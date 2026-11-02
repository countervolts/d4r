#!/usr/bin/env python3
"""Compare RR backends against the same NVIDIA/ZLUDA reference.

Fixed peak 1, unclipped linear RGB, alpha excluded. Pooled MSE is aggregated
before converting to dB; per-frame results are retained. Missing/misshapen or
nonfinite frames fail. --max-extra-loss controls the D3D12-vs-Linux quality gap,
not image agreement between those backends.
"""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def read(path,pixels):
    a=np.fromfile(path,np.float16)
    if a.size!=pixels*4:raise ValueError(f'{path}: wrong shape ({a.size} values)')
    a=a.reshape(-1,4)[:,:3].astype(np.float64)
    if not np.isfinite(a).all():raise ValueError(f'{path}: nonfinite RGB')
    return a


def psnr(mse):return -10*math.log10(mse) if mse else math.inf


def compare(case,reference,linux,d3d12,max_extra_loss=2):
    rows=[l.split() for l in (case/'frames.txt').read_text().splitlines() if l and not l.startswith('#')]
    if rows[0][0]!='size':raise ValueError('no size')
    pixels=int(rows[0][3])*int(rows[0][4]);frames=[int(r[1]) for r in rows[1:] if r[0]=='frame']
    if not frames:raise ValueError('empty case')
    result={'reference':'NVIDIA CUDA through ZLUDA','metric':'fixed peak 1, unclipped linear RGB, pooled MSE; alpha excluded','frames':[],'backends':{}}
    sums={'linux':0.,'d3d12':0.}
    for f in frames:
        ref=read(reference/f'{f}-out.bin',pixels);row={'frame':f}
        for label,folder in [('linux',linux),('d3d12',d3d12)]:
            a=read(folder/f'{f}-out.bin',pixels);delta=a-ref;mse=float(np.mean(delta*delta));sums[label]+=mse
            row[label]={'psnr_db':psnr(mse),'mse':mse,'max_abs_error':float(np.max(np.abs(delta)))}
        result['frames'].append(row)
    for label,value in sums.items():result['backends'][label]={'pooled_psnr_db':psnr(value/len(frames)),'pooled_mse':value/len(frames)}
    extra=result['backends']['linux']['pooled_psnr_db']-result['backends']['d3d12']['pooled_psnr_db']
    result['extra_loss_db']=0 if math.isnan(extra) else extra
    result['max_extra_loss_db']=max_extra_loss;result['passed']=result['extra_loss_db']<=max_extra_loss
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('case','reference','linux','d3d12','output'):p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--max-extra-loss',type=float,default=2);a=p.parse_args()
    result=compare(a.case,a.reference,a.linux,a.d3d12,a.max_extra_loss)
    # JSON has no infinity; exact agreement is recorded explicitly.
    def clean(x):
        if isinstance(x,float) and not math.isfinite(x):return 'infinity' if x>0 else '-infinity'
        if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
        if isinstance(x,list):return [clean(v) for v in x]
        return x
    a.output.write_text(json.dumps(clean(result),indent=2)+'\n')
    print(json.dumps(clean(result['backends']),indent=2));print('extra loss:',result['extra_loss_db'],'dB; passed:',result['passed'])
    raise SystemExit(0 if result['passed'] else 1)
