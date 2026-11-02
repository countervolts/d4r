import re,sys,json,os
M=sys.argv[1]; ow,oh=int(sys.argv[2]),int(sys.argv[3])
w0=(ow+31)&~31; h0=(oh+31)&~31
lv={'enc0':0,'enc1':1,'enc2':2,'enc3':3,'enc4':4,'enc5':5,'dec4':4,'dec3':3,'dec2':2,'dec1':1,'dec0':0}
out={}
for s,l in lv.items():
    d=f"{M}/native-work/{s}"
    h=[f for f in ("swin_stage.h","dec0_m_w.h","enc0_m_w.h") if os.path.exists(f"{d}/{f}")]
    if not h: continue
    txt=open(f"{d}/{h[0]}").read()
    per=0.0; attn=0.0
    for name,kind,r,c in re.findall(r"#define (W_\w+) \d+u\s+// ([AB]) \((\d+), (\d+)\)",txt):
        f=2*int(r)*int(c)
        if re.match(r"W_W[KV]\d",name): f*=1.5      # K,V for 96 keys per 64 queries
        per+=f
    for _ in re.findall(r"#define W_BIAS\d+ ",txt):
        nh=int((re.search(r"#define NH (\d+)",txt) or [0,1])[1]) if "NH" in txt else 1
        attn+=nh*2*2*96*32                          # QK^T and PV per query token, per head, 96 keys, hd 32
    tok=(w0>>l)*(h0>>l)
    out[s]=dict(level=l,tokens=tok,gemm_flop_per_tok=per,attn_flop_per_tok=attn,gflop=(per+attn)*tok/1e9)
json.dump(out,sys.stdout,indent=1)
