#!/usr/bin/env python3
"""Run NVIDIA RR D/E/F through Windows ZLUDA on a native replay case.

Requires locally supplied NGX core/feature DLLs and the built CUDA reference
runner, CUDA bridge and NVAPI shim. Creates a private application directory and leaves game
installations untouched. The saved CUDA outputs, not presented frame age, are
the reference. See engine/d3d12/windows-rr-tuning.md for invocation.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','output','core','dll','sr_dll','zluda'):p.add_argument('--'+name.replace('_','-'),type=Path,required=True)
    p.add_argument('--preset',type=int,choices=[4,5,6],required=True)
    p.add_argument('--runner',type=Path,required=True)
    p.add_argument('--bridge',type=Path,default=ROOT/'build/windows-nvcuda/nvcuda.dll')
    p.add_argument('--nvapi',type=Path,default=ROOT/'build/windows-nvapi/nvapi64.dll')
    a=p.parse_args();a.output=a.output.resolve()
    app=a.output/'app';result=a.output/'reference'
    for d in (app,result):d.mkdir(parents=True,exist_ok=True)
    provenance={}
    for src,name in ((a.core,'_nvngx.dll'),(a.dll,'nvngx_dlssd.dll'),(a.sr_dll,'nvngx_dlss.dll'),(a.bridge,'nvcuda.dll'),(a.runner,'d4r-rr-zluda.exe')):
        shutil.copy2(src,app/name);provenance[name]=hashlib.sha256(src.read_bytes()).hexdigest()
    frame_ids=[int(r[1]) for line in (a.case/'frames.txt').read_text().splitlines() if (r:=line.split()) and r[0]=='frame']
    if not frame_ids:raise ValueError('empty case')
    env={k:v for k,v in os.environ.items() if not k.startswith('D4R_') or k.startswith('D4R_CUDA_')} # the bridge's dump settings pass
    env.update({'D4R_ENGINE':'0','D4R_NGX_CORE':str(app/'_nvngx.dll'),'D4R_NGX_FEATURE_DIR':str(app),
        'D4R_NVAPI':str(a.nvapi.resolve()),'D4R_ZLUDA_LIBCUDA':str(a.zluda.resolve()/'nvcuda.dll'),
        'D4R_RR_ENABLE':'1','D4R_RR_BACKEND':'dlss','D4R_RR_PRESET':str(a.preset),
        'D4R_ZLUDA_NATIVE_DIR':str(a.output/'no-native-kernels'),
        'D4R_ZLUDA_WMMA':'1','D4R_ZLUDA_WMMA_FP8':'1','D4R_ZLUDA_WMMA_FP8_NATIVE':'1',
        'D4R_CUDA_MODULE_DIR':str(a.output/'modules'),'D4R_ELIDE_NGX_SYNC':'0',
        'PATH':str(a.zluda.resolve())+os.pathsep+os.environ['PATH']})
    with (a.output/'runner.log').open('w') as log:
        subprocess.run([str(app/'d4r-rr-zluda.exe'),str(app),str(a.case.resolve()),str(result),str(a.preset)],env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    size=list(map(int,(a.case/'frames.txt').read_text().splitlines()[0].split()[1:]))
    for fid in frame_ids:
        path=result/f'{fid}-out.bin'
        if path.stat().st_size!=size[2]*size[3]*8:raise ValueError('missing/incomplete CUDA reference '+str(path))
    (a.output/'provenance.json').write_text(json.dumps({'preset':a.preset,'dlls':provenance,'frames':frame_ids,'case':str(a.case.resolve()),'reference':'Synchronous NVIDIA CUDA kernels through ZLUDA; no native kernel substitutions'},indent=2)+'\n')
    print(f'{len(frame_ids)} ZLUDA reference frames: {result}')


if __name__=='__main__':main()
