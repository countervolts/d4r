#!/usr/bin/env python3
"""Lossless RR raw case <-> Windows harness replay conversion.

to-shim CASE_DIR REPLAY_DIR writes D4R_HARNESS_REPLAY_DIR inputs for a ZLUDA
reference run. from-shim CAPTURE_DIR CASE_DIR imports the shim's input/output
dumps for d4r-rr and d4r-rr-d3d12. No inferred guides or camera matrices.
"""
import argparse
import json
import shutil
from pathlib import Path

GUIDES=[('normals','GBuffer.Normals',10,8),('roughness','GBuffer.Roughness',10,8),
        ('diffuse','DLSS.Input.DiffuseAlbedo',28,4),('specular','DLSS.Input.SpecularAlbedo',28,4),
        ('hitdistance','DLSSD.SpecularHitDistance',41,4),('disocclusion','DLSS.DisocclusionMask',54,2),
        ('sss','DLSSD.ScreenSpaceSubsurfaceScatteringGuide',54,2),('particles','DLSSD.ColorBeforeParticles',10,8)]
PLANES={'color':'color.rgba16f','motion':'motion.rg16f','depth':'depth.r32f'}


def copy(src,dst,size):
    if src.stat().st_size!=size:raise ValueError(f'{src}: expected {size} bytes')
    shutil.copyfile(src,dst)


def to_shim(src,dst,preset):
    rows=[l.split() for l in (src/'frames.txt').read_text().splitlines() if l and not l.startswith('#')]
    if rows[0][0]!='size':raise ValueError('missing size')
    rw,rh,ow,oh=map(int,rows[0][1:]);refs=[]
    for n,row in enumerate(rows[1:],1):
        if row[0]!='frame' or len(row)!=41:raise ValueError('invalid frame')
        fid=int(row[1]);flag=int(row[8]);prefix=dst/f'frame-{n:06d}'
        params=[f'jitter {row[2]} {row[3]}',f'mv_scale {row[4]} {row[5]}','sharpness 0',f'pre_exposure {row[6]}',
                'exposure_scale 1','frame_time 16.6666679',f'reset {row[7]}','invert 0 0',f'render {rw} {rh}',
                'color_base 0 0','depth_base 0 0','mv_base 0 0','output_base 0 0','has_exposure 0',
                f'create {rw} {rh} {ow} {oh} 2 {3 | (8 if flag&1 else 0)} 0 {preset}']
        (Path(str(prefix)+'-params.txt')).write_text('\n'.join(params)+'\n')
        for name,suffix in PLANES.items():copy(src/f'{fid}-{name}.bin',Path(str(prefix)+'-'+suffix),rw*rh*(8 if name=='color' else 4))
        guides=[]
        for name,ngx,fmt,bpp in GUIDES:
            path=src/f'{fid}-{name}.bin'
            if not path.exists():
                if name in ('normals','roughness','diffuse','specular') and not (name=='roughness' and preset==6 and flag&4):raise ValueError('missing '+name)
                continue
            i=len(guides);copy(path,Path(str(prefix)+f'-guide-{i:02d}.bin'),rw*rh*bpp)
            guides.append(f'guide {i} {ngx} {ngx} {fmt} {rw} {rh} {bpp} 0')
        guides=[f'guides {len(guides)}',*guides,f'scalar DLSS.Roughness.Mode {1 if flag&4 else 0}',f'scalar DLSS.Use.HW.Depth {1 if flag&2 else 0}',
                'matrix WorldToViewMatrix 1 '+' '.join(row[9:25]),'matrix ViewToClipMatrix 1 '+' '.join(row[25:41])]
        Path(str(prefix)+'-guides.txt').write_text('\n'.join(guides)+'\n')
        refs.append({'frame':fid,'shim_frame':n})
    (dst/'mapping.json').write_text(json.dumps(refs,indent=2)+'\n')


def from_shim(src,dst):
    lines=[];refs=[];dims=None;preset=None
    for params in sorted(src.glob('frame-*-params.txt')):
        n=int(params.name.split('-')[1]);prefix=str(params)[:-11]
        fields={r[0]:r[1:] for l in params.read_text().splitlines() if (r:=l.split())}
        c=list(map(int,fields['create']));rw,rh,ow,oh=c[:4]
        if len(c)!=8 or c[7] not in (4,5,6) or c[5]&3!=3 or c[5]&4 or c[6]:raise ValueError('requires HDR RR D/E/F with unjittered render motion and no output subrect')
        if preset is not None and preset!=c[7]:raise ValueError('preset change requires separate RR cases')
        preset=c[7]
        if list(map(int,fields['render']))!=[rw,rh]:raise ValueError('render-size change requires separate RR cases')
        if dims and dims!=(rw,rh,ow,oh):raise ValueError('resize requires separate RR cases')
        dims=(rw,rh,ow,oh)
        if not lines:lines=[f'size {rw} {rh} {ow} {oh}']
        if any(any(int(x) for x in fields[k]) for k in ('color_base','depth_base','mv_base','output_base')):raise ValueError('nonzero origins require cropping before import')
        guides=[l.split() for l in Path(prefix+'-guides.txt').read_text().splitlines()]
        scalars={r[1]:int(r[2]) for r in guides if r[0]=='scalar'}
        matrices={r[1]:r[3:] for r in guides if r[0]=='matrix' and r[2]=='1'}
        flag=(1 if c[5]&8 else 0)|(2 if scalars.get('DLSS.Use.HW.Depth') else 0)|(4 if scalars.get('DLSS.Roughness.Mode') else 0)
        for name,suffix in PLANES.items():copy(Path(prefix+'-'+suffix),dst/f'{n}-{name}.bin',rw*rh*(8 if name=='color' else 4))
        for r in guides:
            if r[0]!='guide':continue
            spec=next((g for g in GUIDES if g[1] in r[2:4] or (g[0]=='diffuse' and 'GBuffer.DiffuseAlbedo' in r[2:4]) or (g[0]=='specular' and 'GBuffer.SpecularAlbedo' in r[2:4])),None)
            if spec is None:raise ValueError('unsupported guide '+r[2])
            name,ngx,fmt,bpp=spec
            if list(map(int,r[4:8]))!=[fmt,rw,rh,bpp]:raise ValueError('guide format/extent conversion required: '+r[2])
            copy(Path(prefix+f'-guide-{int(r[1]):02d}.bin'),dst/f'{n}-{name}.bin',rw*rh*bpp)
        for name in ('normals','roughness','diffuse','specular'):
            if not (dst/f'{n}-{name}.bin').exists() and not (name=='roughness' and preset==6 and flag&4):raise ValueError('missing required guide '+name)
        view=matrices['WorldToViewMatrix'];clip=matrices['ViewToClipMatrix']
        if len(view)!=16 or len(clip)!=16:raise ValueError('incomplete camera')
        lines.append(' '.join(['frame',str(n),*fields['jitter'],*fields['mv_scale'],fields['pre_exposure'][0],fields['reset'][0],str(flag),*view,*clip]))
        reference=Path(prefix+'.rgba16f')
        if reference.exists():copy(reference,dst/f'{n}-reference.bin',ow*oh*8);refs.append(n)
    if not dims:raise ValueError('no captured frames')
    (dst/'frames.txt').write_text('\n'.join(lines)+'\n');(dst/'references.json').write_text(json.dumps(refs)+'\n')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=['to-shim','from-shim']);p.add_argument('source',type=Path);p.add_argument('output',type=Path);p.add_argument('--preset',type=int,choices=[4,5,6],default=4);a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
    if a.mode=='to-shim':to_shim(a.source,a.output,a.preset)
    else:from_shim(a.source,a.output)
