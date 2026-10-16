#!/usr/bin/env python3
# usage: kprof.py RUN.log FRAMES [WARM] : per-kernel median GPU ms and per-frame total from KERNEL_PROFILE lines
import sys,re,statistics as st
lines=[l for l in open(sys.argv[1],errors='replace') if 'KERNEL_PROFILE' in l]
frames=int(sys.argv[2]); warm=int(sys.argv[3]) if len(sys.argv)>3 else frames//2
rows=[(int(m.group(1)),float(m.group(2)),m.group(3)) for m in (re.search(r'launch=(\d+).*gpu_ms=([0-9.]+) name=(\S+)',l) for l in lines) if m]
rows.sort(); n=len(rows); per=n/frames; cut=int(per*warm); rows=rows[cut:]; f=frames-warm
agg={}
for _,ms,name in rows: agg.setdefault(name,[]).append(ms)
tot=sum(sum(v) for v in agg.values())/f
print(f"launches/frame={per:.1f} total_gpu_ms/frame={tot:.3f} (frames {warm+1}-{frames})")
for name,v in sorted(agg.items(),key=lambda kv:-sum(kv[1])):
    print(f"  {sum(v)/f:8.3f} ms/frame  {len(v)/f:5.1f}x  med {st.median(v):7.3f}  {name}")
