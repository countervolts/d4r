#!/usr/bin/env python3
"""Add native D3D12/AGS shaders to an RR native Vulkan model.

Usage: compile_rr_dxil.py MODEL_DIR [--output DIR] [--dxc DXC] [--d3d12-target windows|vkd3d]
D/E (D4RR0001): the model is built with rr/build_native_model.sh (relaxed gfx12
arithmetic). Reuses its preprocessed sources when supplied by the Linux handoff;
otherwise preprocesses the native stage sources and their generated weight
headers. Private weights and generated code remain in the user's model directory.
F (D4RR0002): the model is built by compile_rr2.py; stages, four-phase shaders
and exposures use preprocessed handoff sources or the generated model sources.
D/E/F use the relaxed arithmetic of rr_hlsl.relaxed_math (quality:
windows-rr-tuning.md).
--d3d12-target vkd3d builds for vkd3d-proton with patches/vkd3d-proton (RADV): the
network stages reach the AGS matrices' scalar registers through AGS element
operations (d3d12/ags_wmma.hlsli D4R_AGS_SCALAR_REGISTERS), same arithmetic.
"""
import argparse
import json
import hashlib
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hlsl_build
from d3d12.rr_hlsl import translate, vectorize_attention_bias

HERE = Path(__file__).resolve().parent
PARAM_BYTES = dict(zip(('enc0','enc1','enc2','enc3','enc4','enc5','dec4','dec3','dec2','dec1','dec0','hkpn'),
                      (1032,144,144,144,144,144,144,144,144,144,152,480)))
# F stages whose translated source is the per-pixel reconstruction: the
# Vulkan build of these is unguarded (no `precise`), so they use the same
# relaxation (engine/rr/compile_rr2.py D4R_RR2_UNGUARDED).
F_UNGUARDED = frozenset(('post', 'down', 'pooled', 'downp'))


def dxil_inputs(hlsl):
    return [hlsl, HERE/'d3d12/rr_wmma.hlsli', HERE/'d3d12/m_wmma.hlsli',
            HERE/'d3d12/ags_wmma.hlsli', HERE/'dialect_hlsl.h']


# dxc runs single-threaded and nothing reads a DXIL before the build ends: the
# compiles are queued on a pool and build() waits for them (D4R_DXC_JOBS).
_POOL = ThreadPoolExecutor(int(os.environ.get('D4R_DXC_JOBS') or os.cpu_count() or 4))
_JOBS = []
_TARGET_DEFINES = []    # build()'s --d3d12-target
# DXC's optimization level: -O2 for both targets. The driver optimizes again: on
# AMD's D3D12 driver -O3 gave bit-identical output and the same GPU time (D, E
# and F within 0.3%, RX 9070 XT) for 18 times the compile (D's dec4 164 s
# against 5 s); for vkd3d-proton (RADV) D's dec3 took more than an hour at -O3
_OPT = ['-O2']


def compile_dxil(hlsl, output, compiler, wave, name=None):
    """Queue one HLSL file's compile, skipping an unchanged result (content stamp)."""
    _JOBS.append(_POOL.submit(_compile_dxil, hlsl, output, compiler, wave, name))


def _compile_dxil(hlsl, output, compiler, wave, name):
    name = name or hlsl.stem
    # The emitted HLSL already carries the translator's output; the key is its
    # text, the headers and the flags that change code generation (not the
    # Python scripts, whose unrelated edits would rebuild every shader).
    key = f'cs_6_6|d4r_entry|2021|16bit|{_OPT[0][1:]}|M_EXACT=1|wave={wave}|{" ".join(_TARGET_DEFINES)}|'.encode()
    digest = hashlib.sha256(key + b''.join(p.read_bytes() for p in dxil_inputs(hlsl))).hexdigest()
    stamp = output/(name+'.dxil.sha256')
    if stamp.exists() and stamp.read_text()==digest and (output/(name+'.dxil')).exists():
        return
    cmd=[compiler,'-nologo','-T','cs_6_6','-E','d4r_entry','-HV','2021','-enable-16bit-types',_OPT[0],
         '-I',str(HERE),'-I',str(HERE/'d3d12'),f'-DD4R_WAVE_SIZE={wave}','-DM_EXACT=1',*_TARGET_DEFINES,
         str(hlsl),'-Fo',str(output/(name+'.dxil'))]
    subprocess.run(cmd,check=True)
    stamp.write_text(digest)


def copy_model(model, lines, f_model=False):
    """The files the runtime reads, for --output pointed elsewhere."""
    needed=set()
    for line in lines:
        f=line.split()
        if not f:continue
        if f[0]=='stage':
            if len(f)>4 and f[4]!='-':needed.add(f[4])
            if f_model and len(f)>5 and f[5]!='-':needed.add(f[5])
        elif f[0]=='template':needed.add(f[2])
    needed.add('rr.txt')
    if (model/'enc0-aux.bin').exists():needed.add('enc0-aux.bin')
    return needed


def wide_offset(path):
    """W_WIDE of a gfx11 native weight file (rr/widen_weights.py appended the widened copy): the original size rounded
    up to 512 bytes, halved; the file is that rounded size plus twice the original."""
    total=Path(path).stat().st_size
    for size in range(max(0,total//3-512),total//3+512):
        base=(size+511)//512*512
        if base+2*size==total:return base//2
    raise ValueError(f'{path}: not a gfx11 (widened) RR weight file')


def build_de(model, output, lines, compiler, gfx11=False):
    """gfx11: an RDNA3 model (build_native_model.sh with D4R_RR_GFX11=1, exact arithmetic): the stages' RR_GFX11
    sources on f16 WMMA (rr_hlsl.translate gfx11), no FP8 instruction anywhere."""
    preset=int(next(l.split()[1] for l in lines if l.startswith('preset ')))
    if preset not in (4,5):raise ValueError('requires RR D or E')
    if output!=model:
        for name in copy_model(model, lines):shutil.copy2(model/name, output/name)
    scratch=[]
    for line in lines:
        f=line.split()
        if not f or f[0]!='stage':continue
        name,shader,wave=f[1],Path(f[2]).stem,int(f[3])
        if name!='hkpn' and shader!=name+'_m':raise ValueError('model must use native network stages')
        pre=model/(shader+'.preprocessed.comp')
        if pre.exists() and not gfx11:source=pre.read_text()
        else:
            work=model/'native-work'/name
            src=(work/'enc0_m.comp') if name=='enc0' else (model/'hkpn.generated.comp') if name=='hkpn' else HERE/'rr'/('dec0_m.comp' if name=='dec0' else 'swin_m.comp')
            defs=['-DRR_EXACT_CONVERT','-DRR_RELAXED=2','-DRR_NATIVE']+(['-DRR_PRESET_D'] if preset==4 else [])
            if gfx11:
                # build_native_model.sh's RDNA3 defines; the output stage (translated) takes RR_GFX11 alone, as
                # compile_rr.py compiles it
                defs=['-DRR_GFX11'] if name=='hkpn' else ['-DRR_EXACT_CONVERT','-DRR_GFX11',f'-DW_WIDE={wide_offset(model/f[4])}u']+(['-DRR_PRESET_D'] if preset==4 else [])
            exe=shutil.which('glslangValidator') or str(Path(os.environ.get('VULKAN_SDK','C:/VulkanSDK/1.4.341.1'))/'Bin/glslangValidator.exe')
            source=subprocess.check_output([exe,'-E',*defs,'-I'+str(work),'-I'+str(HERE/'rr'),str(src)],text=True)
        if gfx11:
            # f16 weights from the widened copy: no transposed FP8 tiles, no attention-bias copy
            if name!='hkpn':
                file=Path(f[4]).stem+'-dx.bin'
                shutil.copy2(model/f[4],output/file)
                scratch.append(f'weights {name} {file}')
            scratch.append(f'loopzero {name} {PARAM_BYTES[name]}')
            translated=translate(source,0,speed=True,loop_zero_offset=PARAM_BYTES[name],precise=name!='hkpn',gfx11=True)
            hlsl=output/(shader+'.hlsl');hlsl.write_text(translated)
            words=int(re.search(r'#define RR_SPILL_WORDS (\d+)',translated)[1])
            if words:scratch.append(f'scratch {name} {words}')
            compile_dxil(hlsl, output, compiler, wave)
            print(f'RR {chr(64+preset)} {name}: HLSL (gfx11)',flush=True)
            continue
        transpose=0
        if name!='hkpn':
            import numpy as np
            blob=(model/f[4]).read_bytes()
            if len(blob)%256:raise ValueError('unaligned RR weight tiles')
            transpose=len(blob)
            tiled=np.frombuffer(blob,np.uint8).reshape(-1,16,16).transpose(0,2,1).copy().tobytes()
            file=Path(f[4]).stem+'-dx.bin'
            scratch.append(f'weights {name} {file}')
        scratch.append(f'loopzero {name} {PARAM_BYTES[name]}')
        # the output stage alone without `precise` (rr_hlsl.relaxed_math)
        translated=translate(source,transpose,speed=True,loop_zero_offset=PARAM_BYTES[name],precise=name!='hkpn')
        if transpose:
            translated,bias=vectorize_attention_bias(translated,blob)
            (output/file).write_bytes(blob+tiled+bias)
        # vkd3d-proton: decoder 0's head and MLP loops unrolled again, as the Vulkan
        # build's are (RADV compiles both; the rolled form is for AMD's D3D12 compiler).
        # RX 9070 XT, RR D 4K: dec0 1930 -> 1873 us; encoder 0's, unrolled, 2357 -> 2408
        if name=='dec0' and '-DD4R_VKD3D=1' in _TARGET_DEFINES:
            translated=re.sub(r'\[loop\] for \(uint (hd|br) = 0u; \1 < \((\d+)u \+ rrB0\.Load\(\d+u\)\); \+\+ \1\)',
                              r'[unroll] for (uint \1 = 0u; \1 < \2u; ++ \1)',translated)
        hlsl=output/(shader+'.hlsl');hlsl.write_text(translated)
        words=int(re.search(r'#define RR_SPILL_WORDS (\d+)',hlsl.read_text())[1])
        if words:scratch.append(f'scratch {name} {words}')
        compile_dxil(hlsl, output, compiler, wave)
        print(f'RR {chr(64+preset)} {name}: HLSL',flush=True)
    for p in (0,1):hlsl_build.build(HERE/'exposure_k.comp',output/f'exposure_rr{p}.dxil',[f'PASS={p}','MODE=2'],dxc=compiler)
    # gfx11: the runtime accepts the model's `arithmetic gfx11` only with this line (shaders built for it)
    (output/'rr-d3d12.txt').write_text('D4RRDX01\n'+('arithmetic gfx11\n' if gfx11 else '')+'\n'.join(scratch)+'\n',newline='\n')
    (output/'rr-d3d12.json').write_text(json.dumps({'preset':preset,'arithmetic':('gfx11 exact' if os.environ.get('D4R_RR_GFX11_EXACT')=='1' else 'gfx11 relaxed') if gfx11 else 'gfx12 relaxed','builder':'compile_rr_dxil.py','shader_model':'6.6'},indent=2)+'\n')


def glslang_path():
    return shutil.which('glslangValidator') or str(Path(os.environ.get('VULKAN_SDK','C:/VulkanSDK/1.4.341.1'))/'Bin/glslangValidator.exe')


def preprocess_f(model, stem, stage, phase=None):
    """Preprocess a fresh compile_rr2.py model (no handoff preprocessed source).

    Uses the stage's build.json when its recorded paths exist, otherwise the
    same definitions/includes and sources compile_rr2.py uses: the native
    rr2_m.comp/rr2_w.comp plus native-work/rr2_stage.h, the hybrid generated
    enc0/dec0_1/enc0g sources, or a generated output-pass source.
    """
    exe=glslang_path()
    build=model/(stem+'.build.json')
    if build.is_file():
        try:
            cmd=json.loads(build.read_text()).get('preprocess')
            if cmd:
                cmd=list(cmd);cmd[0]=exe
                sources=[a for a in cmd if not a.startswith('-') and a.endswith('.comp')]
                if sources and all(Path(a).is_file() for a in sources):
                    return subprocess.check_output(cmd,text=True)
        except (OSError,ValueError,KeyError,TypeError,subprocess.SubprocessError):
            pass
    work=model/'native-work'/('enc0' if stage=='enc0g' else stage)
    defs=['-DRR_EXACT_CONVERT','-DRR_RELAXED=2','-DRR_NATIVE']
    exact=bool(os.environ.get('D4R_TMP_EXACT_F')) # TMP experiment: the reference's roundings
    if exact:defs=['-DRR_EXACT_CONVERT']
    includes=[]
    if stage=='enc0' and (work/'enc0_m.comp').is_file():
        source=work/'enc0_m.comp';includes=[work,HERE/'rr']
    elif stage=='enc0g' and (work/'enc0_g.comp').is_file():
        source=work/'enc0_g.comp';includes=[work,HERE/'rr']
    elif stage=='dec0_1' and (work/'dec0_1_m.comp').is_file():
        source=work/'dec0_1_m.comp';includes=[work,HERE/'rr']
    elif (work/'rr2_stage.h').is_file():
        # Default compile_rr2 expression: dec0_0 is the one-wave kernel;
        # every other native stage and every phase is rr2_m.comp.
        source=HERE/'rr'/'rr2_m.comp'
        if phase:defs.append(f'-DRR2_PHASE={phase}')
        elif stage=='dec0_0' and not exact and not os.environ.get('D4R_TMP_DEC0_M'):source=HERE/'rr'/'rr2_w.comp'
        includes=[work,HERE/'rr']
    elif (model/(stage+'.generated.comp')).is_file():
        source=model/(stage+'.generated.comp');defs=[]
    else:
        raise ValueError(f'cannot preprocess F stage {stage} ({stem}): no handoff source, build.json, native-work or generated source')
    return subprocess.check_output([exe,'-E',*defs,*[f'-I{p}' for p in includes],str(source)],text=True)


def exposure_f_source(model, p):
    pre=model/f'exposure_f{p}.preprocessed.comp'
    if pre.is_file():return pre.read_text()
    return subprocess.check_output([glslang_path(),'-E',f'-DPASS={p}',str(HERE/'rr'/'exposure_f.comp')],text=True)


# exposure_f.comp PASS 2's 64-bit atomics on the block sums (u2), which rr_hlsl.translate leaves as GLSL calls: the
# relaxed ones as plain interlocked operations, the count's acquire-release with a device memory barrier either side
_EXPOSURE_ATOMICS = '''uint64_t rrAtomicAdd64(RWByteAddressBuffer b, uint at, uint64_t v) { uint64_t o; b.InterlockedAdd64(at, v, o); return o; }
uint64_t rrAtomicAdd64AcqRel(RWByteAddressBuffer b, uint at, uint64_t v)
{
    DeviceMemoryBarrier();
    uint64_t o;
    b.InterlockedAdd64(at, v, o);
    DeviceMemoryBarrier();
    return o;
}
uint64_t rrAtomicExchange64(RWByteAddressBuffer b, uint at, uint64_t v) { uint64_t o; b.InterlockedExchange64(at, v, o); return o; }
'''


def exposure_one_pass(source):
    def atomic(m):
        op='Add' if m[1]=='atomicAdd' else 'Exchange'
        order='AcqRel' if m[5]=='gl_SemanticsAcquireRelease' else ''
        return f'rrAtomic{op}64{order}(rrB{m[2]}, {int(m[3])*8}u, {m[4]})'
    source,n=re.subn(r'(atomicAdd|atomicExchange)\(rrB(\d+)\.Load<uint64_t>\(\((\d+)\) \* 8u\), ([^,]+), gl_ScopeDevice, [^,]+, (gl_Semantics\w+)\)',atomic,source)
    if n!=4 or 'atomic' in source.replace('rrAtomic',''):raise ValueError('unexpected exposure_f PASS 2 atomics')
    return source.replace('void main()',_EXPOSURE_ATOMICS+'void main()',1)


def build_f(model, output, lines, compiler):
    import numpy as np
    from d3d12 import rr2_native_opt, rr2_pixel_opt
    def optimize(source,stage):
        if os.environ.get('D4R_TMP_EXACT_F'):return source # TMP
        if stage!='enc0g' and stage not in F_UNGUARDED:
            source=rr2_native_opt.optimize(source,stage)
        if stage in ('enc0','dec0_1','enc0g') or stage in F_UNGUARDED:
            source=rr2_pixel_opt.optimize(source,stage)
        return source
    stages={};phases={}
    for line in lines:
        f=line.split()
        if not f:continue
        if f[0]=='stage':stages[f[1]]=f
        elif f[0]=='phases':phases[f[1]]=f
    needed=copy_model(model,lines,True)
    unguarded=F_UNGUARDED
    missing=[n for n in needed if not (model/n).is_file()]
    if not (model/'enc0-aux.bin').is_file():missing.append('enc0-aux.bin')
    if missing:raise ValueError('incomplete F model: missing '+', '.join(sorted(set(missing))))
    if output!=model:
        for name in needed:shutil.copy2(model/name, output/name)
    def source_of(spv, stage, phase=None):
        stem=Path(spv).stem
        pre=model/(stem+'.preprocessed.comp')
        if pre.is_file() and not os.environ.get('D4R_TMP_EXACT_F') and not (os.environ.get('D4R_TMP_DEC0_M') and stage=='dec0_0'):return pre.read_text() # TMP
        return preprocess_f(model, stem, stage, phase)
    def tiled_weights(source, blob):
        # FP8 B operands loaded row-major from the weights need the transposed
        # 16x16 tiles the translator appends (D/E's builder does the same).
        return bool(blob) and len(blob)%256==0 and bool(re.search(
            r'coopMatLoad\s*\([^;]*\bw8\.w\b[^;]*gl_CooperativeMatrixLayoutRowMajor',source))
    scratch=[];written=0;exposure=[]
    for name,f in stages.items():
        shader=f[2];wave=int(f[3]);weight=f[4];argfile=f[5]
        args=(model/argfile).stat().st_size
        phased=bool(phases.get(name))
        # The phased stages' Vulkan ABI writes the phase records' arena address
        # at byte offset args (word 26); the rolled-loop guard must read a zero
        # word after it, which the runtime appends (and the sidecar declares).
        guard=args+4 if phased else args
        source=source_of(shader,name)
        blob=(model/weight).read_bytes() if weight!='-' else b''
        tiled=tiled_weights(source,blob)
        if tiled and len(blob)%256:raise ValueError('unaligned F weight tiles')
        wb=len(blob) if tiled else 0
        hlsl=output/(Path(shader).stem+'.hlsl')
        translated=translate(source,wb,speed=True,loop_zero_offset=guard,
                             precise=name not in unguarded,
                             param_groups=1 if ('gl_NumWorkGroups.x' in source or 'gl_NumWorkGroups.y' in source) else False)
        weight_file=None
        if weight!='-':
            weight_file=Path(weight).stem+'-dx.bin'
            if tiled:
                transposed=np.frombuffer(blob,np.uint8).reshape(-1,16,16).transpose(0,2,1).copy().tobytes()
                translated,bias=vectorize_attention_bias(translated,blob)
                (output/weight_file).write_bytes(blob+transposed+bias)
            else:(output/weight_file).write_bytes(blob)
            scratch.append(f'weights {name} {weight_file}')
        translated=optimize(translated,name)
        hlsl.write_text(translated)
        words=int(re.search(r'#define RR_SPILL_WORDS (\d+)',hlsl.read_text())[1])
        if words:scratch.append(f'scratch {name} {words}')
        # the four-phase shaders share the stage's weights, arguments and regions
        loads_zero=bool(re.search(rf'rrB0\.Load\({guard}u\)',translated))
        info=phases.get(name)
        for pshader in (info[2:6] if info else []):
            phlsl=output/(Path(pshader).stem+'.hlsl')
            phase=int(Path(pshader).stem.rsplit('_p',1)[1])
            scale=int(info[6]) if phase==1 else int(info[7]) if phase==3 else 1
            psrc=source_of(pshader,name,phase)
            ptrans=translate(psrc,wb,speed=True,loop_zero_offset=guard,precise=True,
                             param_groups=scale if ('gl_NumWorkGroups.x' in psrc or 'gl_NumWorkGroups.y' in psrc) else False)
            if tiled:ptrans,_=vectorize_attention_bias(ptrans,blob)
            ptrans=optimize(ptrans,Path(pshader).stem)
            phlsl.write_text(ptrans)
            loads_zero=loads_zero or bool(re.search(rf'rrB0\.Load\({guard}u\)',ptrans))
            compile_dxil(phlsl,output,compiler,wave)
        if loads_zero:scratch.append(f'loopzero {name} {guard}')
        compile_dxil(hlsl, output, compiler, wave)
        written+=1
        print(f'RR F {name}: HLSL' + (', four phases' if name in phases and phases[name] else ''),flush=True)
    for p in (0,1):
        hlsl=output/f'exposure_f{p}.hlsl'
        # The exposure passes are tiny and their pipelines do not bind the AGS
        # intrinsic; dialect_hlsl.h's software d4rRne16 rounds to nearest even
        # as the Vulkan build does (no truncating D3D12 cast).
        hlsl.write_text(translate(exposure_f_source(model,p),0,speed=False,precise=True))
        compile_dxil(hlsl,output,compiler,64)
        exposure.append(f'exposure_f{p}.dxil')
    if '-DD4R_VKD3D=1' in _TARGET_DEFINES:
        # vkd3d-proton: both passes in one with 64-bit atomics, the last block finishing, as the Vulkan build's
        # exposure_f2.spv; the second pass's single workgroup cost as much as the first pass's grid
        hlsl=output/'exposure_f2.hlsl'
        hlsl.write_text(exposure_one_pass(translate(exposure_f_source(model,2),0,speed=False,precise=True)))
        compile_dxil(hlsl,output,compiler,64)
        exposure.append('exposure_f2.dxil')
    (output/'rr-d3d12.txt').write_text('D4RRDX01\n'+'\n'.join(scratch)+'\n',newline='\n')
    (output/'rr-d3d12.json').write_text(json.dumps({'preset':6,'format':'D4RR0002','arithmetic':'gfx12 relaxed',
        'builder':'compile_rr_dxil.py','shader_model':'6.6','stages':list(stages),
        'phases':{k:{'shaders':v[2:6],'heads':int(v[6]),'branches':int(v[7]),'words':int(v[8])} for k,v in phases.items()},'exposure':exposure},indent=2)+'\n')
    print(f'RR F: {written} stages, {len(phases)} four-phase stages',flush=True)


def build(model, output=None, dxc=None, target='windows'):
    if target not in ('windows','vkd3d'):raise ValueError('D3D12 target must be windows or vkd3d')
    _TARGET_DEFINES[:]=['-DD4R_AGS_SCALAR_REGISTERS=1','-DD4R_VKD3D=1'] if target=='vkd3d' else []
    model=Path(model).resolve();output=Path(output).resolve() if output else model
    output.mkdir(parents=True,exist_ok=True)
    lines=(model/'rr.txt').read_text().splitlines()
    if not lines:raise ValueError('empty RR model')
    compiler=hlsl_build.dxc_path(dxc)
    if lines[0]=='D4RR0002':
        if 'arithmetic gfx11' in lines:raise ValueError('requires a gfx12 RR F model')
        build_f(model,output,lines,compiler)
    elif lines[0]=='D4RR0001':
        gfx11='arithmetic gfx11' in lines
        if gfx11 and target=='vkd3d':raise ValueError('a gfx11 RR D/E model is for AMD\'s Windows driver (RADV runs its Vulkan shaders)')
        build_de(model,output,lines,compiler,gfx11)
    else:raise ValueError('unknown RR model format: '+lines[0])
    for job in _JOBS:job.result()
    _JOBS.clear()


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('model',type=Path)
    ap.add_argument('--output',type=Path)
    ap.add_argument('--dxc')
    ap.add_argument('--d3d12-target',choices=('windows','vkd3d'),default='windows',help='AMD\'s Windows driver, or '
                    'vkd3d-proton with patches/vkd3d-proton')
    a=ap.parse_args()
    build(a.model,a.output,a.dxc,a.d3d12_target)
