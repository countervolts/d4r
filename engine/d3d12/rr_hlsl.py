"""Lower the RR native model's preprocessed GLSL to HLSL/AGS.

The standard D/E builder keeps the Linux stage bodies and packed weights,
relaxes arithmetic and keeps FP8 scalar values packed. Unknown matrix layouts
are errors.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hlsl_build


def closing(s, start):
    pairs = {'(': ')', '[': ']', '{': '}'}
    stack = [pairs[s[start]]]
    for i in range(start + 1, len(s)):
        if s[i] in pairs: stack.append(pairs[s[i]])
        elif s[i] == stack[-1]:
            stack.pop()
            if not stack: return i
    raise ValueError('unbalanced shader expression')


def args(s):
    result, at, i = [], 0, 0
    while i < len(s):
        if s.startswith('coopmat',i): i=s.index('>',i)
        elif s[i] in '([{': i = closing(s, i)
        elif s[i] == ',': result.append(s[at:i].strip()); at = i + 1
        i += 1
    return result + [s[at:].strip()]


def calls(s, name, fn):
    pattern = re.compile(r'\b' + name + r'\s*\(')
    pos = 0
    while m := pattern.search(s, pos):
        start = s.index('(', m.start()); end = closing(s, start)
        replacement = fn(args(s[start+1:end]), m.start())
        s = s[:m.start()] + replacement + s[end+1:]
        pos = m.start() + len(replacement)
    return s


MATRIX = re.compile(r'coopmat\s*<\s*(\w+)\s*,\s*gl_ScopeSubgroup\s*,\s*16\s*,\s*16\s*,\s*gl_MatrixUse(\w+)\s*>')


def byte_names(source):
    """Give FP8 matrix locals unique names, respecting lexical shadowing.

    They become two packed words in HLSL. Keeping all eight codes in separate
    uint registers spills the deep stages even at very small render sizes.
    """
    scopes=[(0,len(source))];stack=[]
    for i,c in enumerate(source):
        if c=='{':stack.append(i)
        elif c=='}' and stack:scopes.append((stack.pop(),i))
    def scope(pos):return min((s for s in scopes if s[0]<=pos<=s[1]),key=lambda s:s[1]-s[0])
    declarations=[];names={}
    for m in MATRIX.finditer(source):
        if re.match(r'\s*\(',source[m.end():]):continue
        for item in args(source[m.end():].split(';',1)[0]):
            var=re.match(r'\s*(\w+)((?:\s*\[[^]]+\])*)',item)
            if not var:continue
            rank=var[2].count('[');name=var[1]
            new=f'rr_byte_{name}_{len(names)}' if m[1]=='floate4m3_t' else None
            if new:names[new]=rank
            declarations.append((m.start(),scope(m.start()),name,new))
    candidates={d[2] for d in declarations if d[3]}
    # Scalar locals can shadow a matrix name in a nested block.
    for m in re.finditer(r'\b(?:uint|int|float|float16_t|uvec[234]|ivec[234]|vec[234]|f16vec[234]|bool)\s+(\w+)\b',source):
        if m[1] in candidates:declarations.append((m.start(),scope(m.start()),m[1],None))
    edits=[]
    for m in re.finditer(r'\b(?:'+ '|'.join(map(re.escape,sorted(candidates)))+r')\b',source):
        options=[d for d in declarations if d[2]==m[0] and d[0]<m.start() and d[1][0]<=m.start()<=d[1][1]]
        if not options:continue
        chosen=min(options,key=lambda d:(d[1][1]-d[1][0],-d[0]))
        if chosen[3]:edits.append((m.start(),m.end(),chosen[3]))
    for a,b,v in reversed(edits):source=source[:a]+v+source[b:]
    return source,names


def byte_accesses(source,names):
    for name,rank in names.items():
        pos=0;pattern=re.compile(r'\b'+name+r'\b')
        while m:=pattern.search(source,pos):
            i=m.end()
            for dim in range(rank):
                while i<len(source) and source[i].isspace():i+=1
                if source[i]!='[':break
                i=closing(source,i)+1
            else:
                a=m.start();j=i
                while j<len(source) and source[j].isspace():j+=1
                while j<len(source) and source[j]==')':
                    before=a-1
                    while before>=0 and source[before].isspace():before-=1
                    if before<0 or source[before]!='(':break
                    a=before;j+=1
                    while j<len(source) and source[j].isspace():j+=1
                if j<len(source) and source[j]=='[':
                    end=closing(source,j);index=source[j+1:end];base=source[a:j].strip()
                    after=re.match(r'\s*=\s*(?!=)',source[end+1:])
                    if after:
                        stop=source.index(';',end+1);value=source[end+1+after.end():stop]
                        replacement=f'rrSet({base}, {index}, {value})';source=source[:a]+replacement+source[stop:]
                    else:
                        replacement=f'rrGet({base}, {index})';source=source[:a]+replacement+source[end+1:]
                    pos=a+len(replacement);continue
            pos=m.end()
    return source


def quantize_matrices(source):
    # Recognise the native Q8PAIRS / ACTQ8 expansions, leaving scalar feature
    # quantization in place. One AGS vector conversion instead of eight codecs.
    pattern=re.compile(r'\[\[unroll\]\]\s*for\s*\(int i_\s*=\s*0;\s*i_\s*<\s*8;\s*i_\s*\+=\s*2\)\s*\{')
    pos=0
    while m:=pattern.search(source,pos):
        start=source.index('{',m.start());end=closing(source,start);body=source[start+1:end]
        src=re.search(r'const vec2 [vx]_\s*=\s*vec2\(\s*\((.+?)\)\[i_\]',body)
        dst=re.search(r'\((rr_byte_\w+(?:\[[^]]+\])*)\)\[i_\]\s*=\s*t_\.x',body)
        if src and dst:
            value=f'{dst[1]} = rrQuantMatrix({src[1]}, {"true" if "const vec2 y_" in body else "false"});'
            source=source[:m.start()]+value+source[end+1:];pos=m.start()+len(value)
        else:pos=end+1
    return source


def enclosing_loop(source,pos,var):
    """The innermost unrolled `for` over var whose body contains pos: (header start, body start, body end)."""
    best=None
    for m in re.finditer(r'\[\[?unroll\]\]?\s*for\s*\((?:int|uint)\s+'+var+r'\s*=\s*0u?;\s*'+var+r'\s*<[^;]+;\s*(?:\+\+\s*'+var+r'|'+var+r'\s*\+=\s*[^)]+)\)\s*',source[:pos]):
        at=m.end()
        end=closing(source,at) if source[at]=='{' else source.index(';',at)
        if at<=pos<=end:best=(m.start(),at,end)
    return best


def fp8_batches(source):
    """Per-element FP8 encodes as hardware conversions of eight (or four) values.

    The relaxed GLSL rounds each scalar with the gfx12 conversion. Here a scalar
    is a software codec; one matrix conversion rounds eight lanes' registers at
    once (accumulator and B layouts agree per lane, so element i stays i).
    """
    # A local floate4m3_t[8] filled by per-element saturatedConvertEXT, then
    # copied into a matrix and packed for the arena: keep the raw vec2 values
    # (the copy's rrQuantSet below is the one hardware FP8 encode) and take
    # the arena words from that matrix instead of re-encoding with pack4.
    header=re.compile(r'\[\[?unroll\]\]?\s*for\s*\(\s*int\s+i\s*=\s*0;\s*i\s*<\s*8;\s*i\s*\+=\s*2\s*\)\s*\{')
    pos=0
    while m:=header.search(source,pos):
        start=source.index('{',m.start());end=closing(source,start);body=source[start+1:end]
        q=re.search(r'fe4m3vec2\s+(\w+)\s*;\s*saturatedConvertEXT\(\1\s*,',body)
        y=re.search(r'const vec2\s+(\w+)\s*=\s*vec2\(',body)
        if q and y:
            call=re.search(r'saturatedConvertEXT\([^;]+\)',body)
            if call and f'({y[1]}).x' in call[0]:
                a=re.search(r'(\w+)\s*\[\s*i\s*\]\s*=\s*'+re.escape(q[1])+r'\.x\s*;\s*\1\s*\[\s*i\s*\+\s*1\s*\]\s*=\s*'+re.escape(q[1])+r'\.y\s*;',body)
                if a:
                    newbody=body.replace(a[0],f'{a[1]}[i] = ({y[1]}).x; {a[1]}[i + 1] = ({y[1]}).y;')
                    if len(re.findall(r'\b'+re.escape(q[1])+r'\b',newbody))<=2:
                        newbody=re.sub(r'fe4m3vec2\s+'+re.escape(q[1])+r'\s*;\s*saturatedConvertEXT\([^;]+;\s*','',newbody)
                    source=source[:start+1]+newbody+source[end:]
                    pos=start+1+len(newbody);continue
        pos=end+1
    # matrix <- local array copies, for the pack4 rewrite after the staging
    local_matrix={}
    for mm in re.finditer(r'rrSet\(\s*\(?\s*(rr_byte_\w+(?:\s*\[[^]]+\])*)\s*\)?\s*,\s*i\s*,\s*(\w+)\s*\[\s*i\s*\]\s*\)',source):
        local_matrix.setdefault(mm[2],mm[1])
    # four codes of a word
    code=r'code\(((?:[^()]|\([^()]*(?:\([^()]*\))?[^()]*\))+?)\)'
    source=re.sub(r'(?<!return )'+code+r' \| \('+code+r' << 8u\) \| \('+code+r' << 16u\) \| \('+code+r' << 24u\)',r'pack4(\1, \2, \3, \4)',source)
    source=re.sub(r'pack4\(q8h\(([^()]+)\), q8h\(([^()]+)\), q8h\(([^()]+)\), q8h\(([^()]+)\)\)',r'rrPack4(\1, \2, \3, \4)',source)
    # c[b] = q8h(n) feeding only the pack of that word
    source=re.sub(r'(floate4m3_t (\w+)\[4\];((?:(?!pack4)[\s\S])*?)\2\[b\] = )q8h\((\w+)\);(\s*\}\s*[^;]*?= )pack4\(\2\[0\], \2\[1\], \2\[2\], \2\[3\]\)',
                  r'\1\4;\5rrPack4(\2[0], \2[1], \2[2], \2[3])',source)
    source=re.sub(r'floate4m3_t (\w+)\[4\];(?=(?:(?!pack4\()[\s\S])*?rrPack4\(\1\[0\])',r'float16_t \1[4];',source)
    # decodes straight to binary16 (rrDecodeH), not through f32 and back
    pos=0
    while m:=re.compile(r'float16_t\(\s*uintBitsToFloate4m3EXT\(').search(source,pos):
        outer=source.index('(',m.start());inner=source.index('(',outer+1)
        value=source[inner+1:closing(source,inner)]
        replacement=f'rrDecodeH({value})'
        source=source[:m.start()]+replacement+source[closing(source,outer)+1:];pos=m.start()+len(replacement)
    # one byte of a word (rr_wmma.hlsli rrDecodeByteH: vkd3d-proton converts the word once)
    source=re.sub(r'rrDecodeH\(uint8_t\( \((\w+(?:\[[^]]*\])?) >> \(8u \* ([^()]+?)\)\) & 0xffu\)\)',r'rrDecodeByteH(\1, \2)',source)
    # a pair of bytes of a word, low or high half (rr_wmma.hlsli rrE4Pair: vkd3d-proton converts the word once)
    pair=r'((?:[^()?;]|\([^()]*(?:\([^()]*\))?[^()]*\))+?)\s*\?\s*unpack8\((\w+(?:\[[^]]*\])?)\)\.xy\s*:\s*unpack8\(\2\)\.zw'
    # (not e4pair's own body: rr2_native_opt.py replaces that function)
    source=re.sub(r'(?<!f16vec2\()uintBitsToFloate4m3EXT\(\s*'+pair+r'\)',r'rrE4Pair(\2, \1)',source)
    source=re.sub(r'uintBitsToFloate4m3EXT\(\s*\(\s*'+pair+r'\)\[([^]]+)\]\)',r'rrE4PairAt(\2, \1, \3)',source)
    # loops setting every element of a register: stage, convert after (from
    # binary16 directly when every staged value already is one)
    def half(v,at):
        name=re.fullmatch(r'(\w+)(?:\[[^]]*\])*(?:\.[xy])?',v)
        if not name:return False
        decl=None
        for d in re.finditer(r'\b(float16_t|f16vec2|RRHalf|fe4m3vec2|float|vec2|RRFloat)\s+[\w\s,\[\]]*?\b'+name[1]+r'\b\s*[\[=;,]',source[:at]):decl=d
        if decl and decl[1]=='fe4m3vec2':
            src=re.search(r'\b'+name[1]+r'\s*=\s*fe4m3vec2\((\w+)\)',source[decl.start():at])
            return bool(src) and half(src[1],decl.start())
        return bool(decl) and decl[1] in ('float16_t','f16vec2','RRHalf')
    n=0;pos=0
    while m:=re.compile(r'rrSet\(').search(source,pos):
        start=source.index('(',m.start());end=closing(source,start)
        target,index,value=args(source[start+1:end])
        var=re.findall(r'\b([a-z]\w*)\b',index)
        loop=enclosing_loop(source,m.start(),var[0]) if len(var)==1 else None
        if not loop:pos=end;continue
        head,body,stop=loop
        sets=[s for s in re.finditer(r'rrSet\(',source[body:stop+1])]
        names={args(source[body+s.end():closing(source,body+s.end()-1)])[0] for s in sets}
        if names!={target}:pos=end;continue
        staged=f'rrS{n}_';n+=1
        values=[]
        def stage(s):
            def one(a,_):
                t,i,v=a;v=re.sub(r'^q8h\((.*)\)$',r'\1',v.strip());v=re.sub(r'^q8s\((.*)\)$',r'rrInfNan(\1)',v)
                values.append(v);return f'{staged}[{i}] = {v}'
            return calls(s,'rrSet',one)
        inner=stage(source[head:stop+1])
        if all(half(v,m.start()) for v in values):
            inner=re.sub(r'const fe4m3vec2 (\w+) = fe4m3vec2\((\w+)\);',r'const f16vec2 \1 = \2;',inner)
            replacement='{ RRHalf '+staged+'; '+inner+f' {target} = rrQuantSetH({staged}); }}'
        else:replacement='{ RRFloat '+staged+'; '+inner+f' {target} = rrQuantSet({staged}); }}'
        source=source[:head]+replacement+source[stop+1:];pos=head+len(replacement)
    # a register's codes re-packed (decode, encode): the word itself. These
    # words are stored under a condition; a conversion used only there is
    # moved into the divergent branch by the driver, whose lowering then reads
    # inactive lanes. rrPin's unconditional shared store keeps it in uniform flow.
    # (After the loop staging above: the assignment to the register it reads is
    # written by that staging.)
    pinned=set()
    for half,first in (('x',0),('y',4)):
        pattern=re.compile(r'pack4\('+', '.join(r'rrGet\(([\w\[\]]+), '+str(first+i)+r'\)' if i==0 else r'rrGet\(\1, '+str(first+i)+r'\)' for i in range(4))+r'\)')
        while m:=pattern.search(source):
            name=m[1]
            if name not in pinned:
                defs=list(re.finditer(re.escape(name)+r'\s*\)*\s*=(?!=)[^;]*;',source[:m.start()]))
                if not defs:raise ValueError('re-packed register without definition: '+name)
                at=defs[-1].end();source=source[:at]+f' rrPin({name});'+source[at:];pinned.add(name)
                m=pattern.search(source)
            source=source[:m.start()]+f'({name}).{half}'+source[m.end():]
    for a,mat in local_matrix.items():
        for first,comp in ((0,'x'),(4,'y')):
            vals=', '.join(re.escape(f'{a}[{first+k}]') for k in range(4))
            source=re.sub(r'pack4\(\s*'+vals+r'\s*\)',lambda _m,f=f'({mat}).{comp}':f,source)
    return source


def vector_loads(source):
    """Eight consecutive binary16 loads of an unrolled i-loop as two 4-vector loads.

    RADV merges a lane's contiguous half loads; the D3D12 compiler issues them
    one 16-bit load at a time. The element address is base + i (directly or
    through a `const uint k = ... + uint(i)` local); the base is hoisted before
    the loop. Templated loads need only the element's alignment (not Load4's).
    """
    n=0;pos=0
    header=re.compile(r'\[\[?unroll\]\]?\s*for\s*\(int i\s*=\s*0;\s*i\s*<\s*8;\s*\+\+\s*i\)\s*')
    while m:=header.search(source,pos):
        at=m.end();end=closing(source,at) if source[at]=='{' else source.index(';',at)
        body=source[at:end+1]
        locals_={k:v for k,v in re.findall(r'const uint (\w+) = ([^;,]+?\+ uint\(i\));',body)}
        hoist=[]
        def base(x):
            x=x.strip()
            for k,v in locals_.items():
                if re.search(r'\b'+k+r'\b',x):x=re.sub(r'\b'+k+r'\b','('+v+')',x)
            if x.count('uint(i)')!=1 or not re.search(r'\+ uint\(i\)\)*$',x):return None
            # linear only: no call (inter(), ...) around the index
            depth=0
            for j in range(x.index('uint(i)')-1,-1,-1):
                if x[j]==')':depth+=1
                elif x[j]=='(':
                    if depth:depth-=1
                    elif j and (x[j-1].isalnum() or x[j-1]=='_'):return None
            return x.replace('+ uint(i)','+ 0u')
        def load(a,_):
            x=re.fullmatch(r'\((.*)\) \* 2u',a[0].strip())
            b=base(x[1]) if x else None
            if b is None:return 'rrB1.Load<float16_t>('+a[0]+')'
            name=f'rrV{n+len(hoist)}_';hoist.append(f'const RRHalf8 {name} = rrHalf8(rrB1.Load<vector<float16_t, 4> >(({b}) * 2u), rrB1.Load<vector<float16_t, 4> >(({b}) * 2u + 8u));')
            return f'rrHalfAt({name}, uint(i))'
        new=calls(body,r'rrB1\.Load<float16_t>',load)
        if hoist:
            n+=len(hoist)
            replacement='{ '+' '.join(hoist)+' '+source[m.start():at]+new+' }'
            source=source[:m.start()]+replacement+source[end+1:];pos=m.start()+len(replacement)
        else:pos=end+1
    return source




def hoist_weight_loads(source):
    """Issue weight operand loads before the products of their unrolled loops.

    Each AGS operation is a sequence of UAV atomics, which no compiler moves a
    buffer load across: a load written before its product waits the full memory
    latency per product (one round trip per WMMA in the small deep stages).
    A load leaves the unrolled loops whose iterations its address depends on
    (constant locals substituted), as long as everything else in the address
    comes from outside them, into a register array filled by a copy of the loop
    nest just before it: FP8 operands (Load2) and binary16 bias rows.
    """
    # (pattern, element type, load statement, use): group 1 is the destination;
    # the address follows the match up to its closing parenthesis
    forms=[(re.compile(r'\{ \(\((rr_byte_\w+)\)\) = rrB1\.Load2\('),'RRByte',
            lambda w,a:f'{w} = rrB1.Load2({a});',lambda d,w:f'{{ (({d})) = {w}'),
           (re.compile(r'\{ \[unroll\] for\(uint ri_=0;ri_<8;\+\+ri_\) \{ \((\w+)\)\[ri_\] = rrB1\.Load<float16_t>\('),'RRHalf',
            lambda w,a:f'[unroll] for (uint ri_ = 0u; ri_ < 8u; ++ ri_) {w}[ri_] = rrB1.Load<float16_t>({a});',
            lambda d,w:f'{{ [unroll] for(uint ri_=0;ri_<8;++ri_) {{ ({d})[ri_] = {w}[ri_]')]
    def value(x):
        x=re.sub(r'\b(\d+)u\b',r'\1',x)
        if not re.fullmatch(r'[\d\s+*/()-]+',x):raise ValueError(x)
        return eval(x.replace('/','//'))
    loops=[]
    for m in re.finditer(r'\bfor\s*\(',source):
        paren=source.index('(',m.start());close=closing(source,paren)
        body=close+1
        while source[body].isspace():body+=1
        if source[body]!='{':continue
        start=m.start();unrolled=re.search(r'\[unroll\]\s*$',source[:start])
        if unrolled:start=unrolled.start()
        h=re.fullmatch(r'\s*uint (\w+) = ([^;]+);\s*\1 < ([^;]+);\s*(?:\+\+ \1|\1 \+= (\w+))\s*',source[paren+1:close])
        try:loop=dict(start=start,body=body,end=closing(source,body),var=h[1],lo=h[2],hi=h[3],step=h[4] or '1u',
                      trips=(value(h[3])-value(h[2])+value(h[4] or '1u')-1)//value(h[4] or '1u')) if unrolled and h else None
        except ValueError:loop=None
        loops.append(loop or dict(start=start,body=body,end=closing(source,body),var=None))
    def encloses(at,p):
        depth=0
        for ch in source[at:p]:
            depth+=(ch=='{')-(ch=='}')
            if depth<0:return False
        return True
    identifiers=lambda x:set(re.findall(r'\b[A-Za-z_]\w*\b',x))
    groups={};edits=[];n=0
    for pattern,typ,stmt,use in forms:
        for l in pattern.finditer(source):
            p=l.start();open_=l.end()-1;stop=closing(source,open_);address=source[open_+1:stop]
            chain=[]
            for loop in sorted((x for x in loops if x['body']<p<x['end']),key=lambda x:-x['body']):
                if not loop['var']:break
                chain.append(loop)
            best=None;addr=address
            for k,loop in enumerate(chain,1):
                for d in re.finditer(r'\bconst uint ([^;]+);',source[loop['body']:p]):
                    at=loop['body']+d.start()
                    if not encloses(at,p):continue
                    for item in args(d[1]):
                        v=re.fullmatch(r'(\w+) = (.+)',item.strip(),re.S)
                        if v:addr=re.sub(r'\b'+v[1]+r'\b','('+v[2]+')',addr)
                inner=source[loop['body']:loop['end']]
                local=set(re.findall(r'\b(\w+)\s*=(?!=)',inner))|set(re.findall(r'\b(?:uint|int|bool|float|float16_t|RRByte|RRFloat|RRHalf|RRHalf8)\s+(\w+)',inner))
                if not (identifiers(addr)-{x['var'] for x in chain[:k]}-{'ri_','WaveGetLaneIndex'})&local:best=(k,addr)
            if not best:continue
            k,addr=best;nest=chain[:k][::-1]
            name=f'rrW{n}_';n+=1
            element=name+''.join(f"[({x['var']} - ({x['lo']})) / ({x['step']})]" for x in nest)
            headers=' '.join(f"[unroll] for (uint {x['var']} = {x['lo']}; {x['var']} < {x['hi']}; {x['var']} += {x['step']})" for x in nest)
            dims=''.join(f"[{x['trips']}]" for x in nest)
            groups.setdefault(id(nest[0]),(nest[0],[]))[1].append(f'{typ} {name}{dims}; {headers} {{ {stmt(element,addr)} }}')
            edits.append((p,stop+1,use(l[1],element)))
    for loop,items in groups.values():
        edits.append((loop['start'],loop['start'],'{ '+' '.join(items)+' '))
        edits.append((loop['end']+1,loop['end']+1,' }'))
    for a,b,v in sorted(edits,key=lambda e:(e[0],e[1]),reverse=True):source=source[:a]+v+source[b:]
    return source


def vectorize_attention_bias(source, weights):
    """Append exact transposed half tiles for the native 96-token attention bias.

    Each lane originally gathers eight half words at a 64-element row stride.
    The extra copy makes those same eight words contiguous for two vector loads.
    FP8's existing transposed copy precedes this one in the weight buffer.
    """
    pattern = re.compile(r'\[unroll\] for \(uint ri_ = 0u; ri_ < 8u; \+\+ ri_\) '
                         r'(rrW\d+_\[[^;]+?)\[ri_\] = rrB1\.Load<float16_t>\(([^;]+)\);')
    matches = list(pattern.finditer(source))
    if not matches:
        return source, b''
    if len(matches) != 1:
        raise ValueError('ambiguous RR attention bias load')
    match = matches[0]
    base = re.search(r'\(\((\d+)u \+ HEAD \* hd\) / 2u \+ kt \* 1024u \+ 16u \* wv\)', match[2])
    if not base:
        raise ValueError('unknown RR attention bias address')
    base = int(base[1])
    expected = (f'(((({base}u + HEAD * hd) / 2u + kt * 1024u + 16u * wv) + '
                '(ri_ + 8u * (WaveGetLaneIndex() >> 4u)) * (64u) + (WaveGetLaneIndex() & 15u))) * 2u')
    head = re.search(r'const uint HEAD = (\d+)u - (\d+)u', source)
    heads = re.search(r'for \(uint hd = 0u; hd < \((\d+)u? \+ rrB0.Load', source)
    if (match[2] != expected or not head or not heads or
            '#define D4R_THREADS 128' not in source or
            '[unroll] for (uint kt = 0u; kt < 6u; kt += 1u)' not in source[:match.start()]):
        raise ValueError('unsupported RR attention bias layout')
    stride = int(head[1]) - int(head[2])
    heads = int(heads[1])
    if stride < 12288 or heads not in (2, 4, 8) or base + (heads - 1) * stride + 12288 > len(weights):
        raise ValueError('RR attention bias exceeds original weights')
    packed = bytearray()
    for h in range(heads):
        for tile in range(6):
            for wave in range(4):
                for col in range(16):
                    for row in range(16):
                        at = base + h * stride + 2 * (tile * 1024 + row * 64 + wave * 16 + col)
                        packed.extend(weights[at:at + 2])
    offset = (f'{2 * len(weights)}u + ((hd * 24u + kt * 4u + wv) * 256u + '
              '(WaveGetLaneIndex() & 15u) * 16u + (WaveGetLaneIndex() >> 4u) * 8u) * 2u')
    replacement = ('{ const uint rrBiasOffset_ = ' + offset + '; const RRHalf8 rrBias_ = '
                   'rrHalf8(rrB1.Load< vector<float16_t, 4> >(rrBiasOffset_), '
                   'rrB1.Load< vector<float16_t, 4> >(rrBiasOffset_ + 8u)); '
                   '[unroll] for (uint ri_ = 0u; ri_ < 8u; ++ri_) ' + match[1] +
                   '[ri_] = rrHalfAt(rrBias_, ri_); }')
    return source[:match.start()] + replacement + source[match.end():], bytes(packed)


def relaxed_math(source, precise=True):
    """Native half arithmetic for the full-resolution RR stages.

    This arithmetic is approximate; final RGB quality is evaluated separately
    against the CUDA reference, including temporal frames. An f32 clamp before FP8
    conversion follows the relaxed Vulkan path; preserving nonfinite values
    here instead loses quality on the large replay. Conversions to binary16
    still round to nearest even (rr_wmma.hlsli's d4rRne16): truncating ones
    cost preset E 1.7 dB. precise=False drops `precise`: free quality-wise only
    in the output stage (hkpn), where it saves about 0.2 ms at 4K; in the
    network stages it costs nothing and is worth 1.2 dB for E. Resource layouts
    and synchronization remain shared with the baseline.
    """
    for name in ('fma16', 'fma16q'):
        match = re.search(r'float16_t ' + name + r'\([^\n]+\)\s*\{', source)
        if match:
            start = source.index('{', match.start())
            end = closing(source, start)
            source = source[:start] + '{ return mad(a, b, c); }' + source[end + 1:]
    if not precise:
        source = re.sub(r'\bprecise\s+', '', source)
    return '#define D4R_RNE16_EXTERNAL 1\n#define RR_QUANT_CLAMPONLY 1\n' + source


def int_expr(x):
    """A preprocessed GLSL integer expression (buffer bindings, pool extents)."""
    x=x.strip()
    if not re.fullmatch(r'[\d\s+*/()-]+',x):raise ValueError('unknown integer expression: '+x)
    return int(eval(x,{'__builtins__':{}},{}))


def translate(source,weight_bytes=0,relaxed=False,speed=False,loop_zero_offset=None,precise=True,param_groups=False,
              gfx11=False):
    """gfx11: the RR_GFX11 (RDNA3) build of a stage (engine/rr/rr_gfx11.glsl): f16 A/B operands of 16 elements per lane
    and RADV's gfx11 register layouts, which AMD's D3D12 driver gives AGS's WMMA too (A/B: lane l holds row/column
    l & 15, element i = k i; accumulators: lane l holds column l & 15, element i = row 2i + (l >> 4)). No FP8 matrix
    or conversion reaches the GPU (RDNA3 has none). relaxed_math applies as for gfx12 (D4R_RR_GFX11_EXACT=1: not)."""
    if speed:
        if loop_zero_offset is None or loop_zero_offset <= 0 or loop_zero_offset % 4:
            raise ValueError('optimized translation requires the runtime loop-zero word offset')
        source=re.sub(r'(?:\[\[(?:unroll|dont_unroll)\]\]\s*)?(for\s*\(uint\s+(?:hd|br)\b)',r'[[dont_unroll]] \1',source)
    source = re.sub(r'^\s*#line[^\n]*', '', source, flags=re.M)
    # GLSL drivers unroll the three attention jobs. With a dynamic job index
    # DXC spills matrix operand arrays into thread scratch. Keep the explicit
    # rolled head/MLP loops, and unroll these short constant loops only.
    edits=[]
    for m in re.finditer(r'\bfor\s*\((?:uint|int)\s+(\w+)\s*=\s*0u?\s*;\s*\1\s*<\s*(\d+)u?\s*;[^)]+\)',source):
        if int(m[2])<=32 and not re.search(r'\[\[(?:unroll|dont_unroll)\]\]\s*$',source[max(0,m.start()-40):m.start()]):
            edits.append(m.start())
    for at in reversed(edits):source=source[:at]+'[[unroll]] '+source[at:]
    source,byte_vars=byte_names(source)
    if not gfx11:source=quantize_matrices(source)   # FP8 matrices only
    # Matrix role is needed only for load/store addressing. Record the declared
    # variables before removing the type; declarations in these stage bodies
    # consistently use each matrix name with one role/type within its scope.
    declarations = []
    for m in MATRIX.finditer(source):
        tail = source[m.end():]
        if re.match(r'\s*\(', tail): continue  # constructor
        decl = tail.split(';', 1)[0]
        for item in args(decl):
            v = re.match(r'\s*(\w+)', item)
            if v: declarations.append((m.start(), v[1], m[1], m[2]))

    def matrix_type(expr, pos):
        explicit = MATRIX.search(expr)
        if explicit: return explicit[1], explicit[2]
        v = re.search(r'\w+', expr)[0]
        found = [d for d in declarations if d[0] < pos and d[1] == v]
        if not found: raise ValueError(f'unknown matrix {expr}')
        return found[-1][2:]

    # Expand loads/stores before changing offsets in the source (positions above).
    # Storage arrays whose elements are DWORDs: shared uint arrays and uint
    # buffer fields (arena.w). A cooperative FP8/half matrix in them packs
    # several elements per word, and an offset/stride counts words.
    dword=set();dword_buffers={}
    for m in re.finditer(r'layout\s*\([^)]*\)\s*(?:readonly\s+|writeonly\s+)?buffer\s+\w+\s*\{\s*uint\s+(\w+)\s*\[\s*\]\s*;\s*\}\s*(\w+)\s*;',source):
        name=m[2]+'.'+m[1];dword.add(name)
        bind=re.search(r'binding\s*=\s*([^,)]+)',m[0])
        if bind:dword_buffers[name]=int_expr(bind[1])
    for m in re.finditer(r'\b(?:shared)\s+uint\s+(\w+)\s*\[',source):dword.add(m[1])
    edits = []
    for m in re.finditer(r'\bcoopMat(Load|Store)\s*\(', source):
        start = source.index('(', m.start()); end = closing(source, start)
        x, arr, off, stride, layout = args(source[start+1:end])
        typ, role = matrix_type(x, m.start())
        if layout not in ('gl_CooperativeMatrixLayoutRowMajor', 'gl_CooperativeMatrixLayoutColumnMajor'):
            raise ValueError(layout)
        count = 8
        if gfx11:
            if typ == 'floate4m3_t': raise ValueError('an FP8 matrix in a gfx11 stage')
            if role == 'A': row, col, count = '(WaveGetLaneIndex() & 15u)', 'ri_', 16
            elif role == 'B': row, col, count = 'ri_', '(WaveGetLaneIndex() & 15u)', 16
            else: row, col = '(2u * ri_ + (WaveGetLaneIndex() >> 4u))', '(WaveGetLaneIndex() & 15u)'
        else:
            row = '((WaveGetLaneIndex() & 15u))' if role == 'A' else '(ri_ + 8u * (WaveGetLaneIndex() >> 4u))'
            col = '(ri_ + 8u * (WaveGetLaneIndex() >> 4u))' if role == 'A' else '(WaveGetLaneIndex() & 15u)'
        rr, cc = (row, col) if layout.endswith('RowMajor') else (col, row)
        # A storage array's unit is a scalar (weights/f16 shared) or a packed uint.
        packed = arr not in ('w8.w', 'w16.w') and arr in dword
        size = 1 if typ == 'floate4m3_t' else (2 if typ == 'float16_t' else 4)
        # Native packed FP8 operands: a contiguous row/column is two DWORDs.
        # The builder appends transposed weight tiles for B's row-major loads,
        # as M/L's builder does for their input/expansion matrices.
        normal_layout=(role=='A' and layout.endswith('RowMajor')) or (role=='B' and layout.endswith('ColumnMajor'))
        if typ=='floate4m3_t' and ((packed and normal_layout) or (arr=='w8.w' and (normal_layout or weight_bytes))):
            lane='(WaveGetLaneIndex() & 15u)';half='(8u * (WaveGetLaneIndex() >> 4u))'
            if arr=='w8.w':
                extra=weight_bytes if role=='B' and layout.endswith('RowMajor') else 0
                at=f'({off}) + {extra}u + {lane} * ({stride}) + {half}'
                if m[1]!='Load':raise ValueError('write to weights')
                body=f'({x}) = rrB1.Load2({at});'
            else:
                at=f'({off}) + {lane} * ({stride}) + ({half} >> 2u)'
                if arr in dword_buffers:
                    bind=dword_buffers[arr]
                    if m[1]=='Load':body=f'({x}) = rrB{bind}.Load2(({at}) * 4u);'
                    else:body=f'rrB{bind}.Store<uint>(({at}) * 4u, ({x}).x); rrB{bind}.Store<uint>((({at}) + 1u) * 4u, ({x}).y);'
                elif m[1]=='Load':body=f'({x}) = uint2({arr}[{at}], {arr}[({at})+1u]);'
                else:body=f'{arr}[{at}] = ({x}).x; {arr}[({at})+1u] = ({x}).y;'
            edits.append((m.start(),end+1,'{ '+body+' }'))
            continue
        index = f'(({off}) + {rr} * ({stride}) + {cc})'
        if packed: index = f'(({off}) * 4u + ({rr} * ({stride}) * 4u) + {cc} * {size}u)'
        value = f'({x})[ri_]'
        if m[1] == 'Load':
            if packed:
                bits = f'(({arr}[({index}) >> 2u] >> (({index} & 3u) * 8u)) & {255 if size == 1 else 65535}u)'
                read = f'rrDecode({bits})' if size == 1 else f'float16_t(f16tof32({bits}))'
            else: read = f'{arr}[{index}]'
            body = f'{value} = {read};'
        else:
            if packed:
                # A complete packed word has one owner (i=0,4 for FP8, i=0,2,... for f16).
                step = 4 // size
                if 'ri_' in cc:
                    # consecutive elements of this lane are adjacent in memory
                    vals = [f'rrCode(({x})[ri_+{i}])' if size == 1 else f'f32tof16(float(({x})[ri_+{i}]))' for i in range(step)]
                    bits = ' | '.join(f'({v} << {i*size*8}u)' for i, v in enumerate(vals))
                    body = f'if ((ri_ % {step}) == 0) {arr}[({index}) >> 2u] = {bits};'
                elif size == 2:
                    # adjacent in memory is the same element of the neighbouring lane (an
                    # accumulator stored row-major): the lane pair swaps one of elements ri_ and
                    # ri_ + 1, then the even lane stores row ri_ and the odd lane row ri_ + 1
                    odd = '(WaveGetLaneIndex() & 1u)'
                    mine = f'({odd} != 0u ? ({x})[ri_+1] : ({x})[ri_])'
                    sent = f'({odd} != 0u ? ({x})[ri_] : ({x})[ri_+1])'
                    at = index.replace('ri_', f'(ri_ + {odd})')
                    body = (f'if ((ri_ % 2) == 0) {{ const uint rrm_ = f32tof16(float({mine})), '
                            f'rrn_ = f32tof16(float(WaveReadLaneAt({sent}, WaveGetLaneIndex() ^ 1u))); '
                            f'{arr}[({at}) >> 2u] = {odd} != 0u ? (rrn_ | (rrm_ << 16u)) : (rrm_ | (rrn_ << 16u)); }}')
                else:
                    raise ValueError('packed FP8 store across lanes')
            else: body = f'{arr}[{index}] = {value};'
        if gfx11 and packed and size == 4: raise ValueError('an f32 matrix in packed storage (gfx11)')
        edits.append((m.start(), end+1, '{ [unroll] for(uint ri_=0;ri_<' + str(count) + ';++ri_) { ' + body + ' } }'))
    for start,end,value in reversed(edits): source=source[:start]+value+source[end:]
    if gfx11:
        source = MATRIX.sub(lambda m: ('RRHalf16' if m[2] in ('A', 'B') else 'RRHalf') if m[1] == 'float16_t' else
                            'RRFloat' if m[1] == 'float' else 'RRByte', source)
        source = re.sub(r'\bRRHalf16\s*\(', 'rrHalf16(', source)
    source = MATRIX.sub(lambda m: 'RRHalf' if m[1] == 'float16_t' else 'RRByte' if m[1]=='floate4m3_t' else 'RRFloat', source)
    def initialize_bytes(m):
        items=[]
        for item in args(m[1]):
            v=re.fullmatch(r'\s*(\w+)((?:\s*\[[^]]+\])*)\s*',item)
            items.append(item+f' = (RRByte{v[2]})0' if v else item)
        return 'RRByte '+', '.join(items)+';'
    source=re.sub(r'\bRRByte\s+([^;]+);',initialize_bytes,source)
    source = re.sub(r'\b(RRFloat|RRHalf)\s*\(', lambda m: ('rrFloat' if m[1]=='RRFloat' else 'rrHalf')+'(', source)
    source = source.replace('coopMatMulAdd', 'rrMma')
    if gfx11:
        # a matrix's length() (the GLSL loops over its elements): 16 for an f16 operand, 8 for an accumulator
        sizes={}
        for d in re.finditer(r'\b(RRHalf16|RRHalf|RRFloat)\s+([^;(]+);',source):
            for item in args(d[2]):
                v=re.match(r'\s*(\w+)',item)
                if v:sizes[v[1]]=16 if d[1]=='RRHalf16' else 8
        for d in re.finditer(r'\b(RRHalf16|RRHalf|RRFloat)\s+(\w+)\s*=',source):sizes[d[2]]=16 if d[1]=='RRHalf16' else 8
        def length(m):
            if m[1] not in sizes:raise ValueError('length() of an unknown matrix '+m[1])
            return f'{sizes[m[1]]}'
        source=re.sub(r'\b(\w+)\.length\(\)',length,source)
    else:
        source=byte_accesses(source,byte_vars)
        source=fp8_batches(source)
    resources = []
    spill = re.search(r'shared uint POOL\[([^;]+)\];',source)
    # The deep decoder's expansion pool exceeds D3D12's 32 KiB group limit.
    # Give each window its own UAV slice; keep every other shared array in LDS.
    spill_words = 0
    if spill:
        expression=spill[1]
        if not re.fullmatch(r'[\d\s+*/()<>?:u-]+',expression):raise ValueError('unknown pool extent')
        # All handoff stage constants have been preprocessed. Conditional sizes
        # use GLSL's ternary operator; ask the small integer expression evaluator.
        from d3d12.rr_pool import extent
        spill_words=extent(expression)
        if spill_words*4<=28000:spill_words=0
    if spill_words:
        source=source[:spill.start()]+source[spill.end():]
        at=0
        while m:=re.compile(r'\bPOOL\s*\[').search(source,at):
            start=source.index('[',m.start());end=closing(source,start)
            value='rrScratch[rrScratchBase + ('+source[start+1:end]+')]'
            source=source[:m.start()]+value+source[end+1:];at=m.start()+len(value)
        resources += ['RWStructuredBuffer<uint> rrScratch : register(u40);','static uint rrScratchBase;']
    buffers = {}
    # A buffer whose body is a struct of named fields (the exposure passes'
    # Params and Values) is a raw buffer as well: its fields are explicit byte
    # offsets. std430: scalars 4 (8 for uint64_t), vectors 8 or 16 and aligned.
    HLSL_TYPE={'uint':'uint','int':'int','float':'float','uint64_t':'uint64_t','uint16_t':'uint16_t',
               'float16_t':'float16_t','vec2':'float2','vec3':'float3','vec4':'float4',
               'ivec2':'int2','ivec3':'int3','ivec4':'int4','uvec2':'uint2','uvec3':'uint3','uvec4':'uint4'}
    def struct_fields(body):
        off=0;result=[]
        for decl in body.split(';'):
            decl=decl.strip()
            if not decl:continue
            d=re.match(r'(\w+)\s+(.+)',decl)
            if not d:raise ValueError('unsupported RR buffer: '+body)
            typ,rest=d.groups()
            if typ not in HLSL_TYPE:raise ValueError('unsupported RR buffer field: '+decl)
            components={'vec2':2,'ivec2':2,'uvec2':2,'vec3':3,'ivec3':3,'uvec3':3,'vec4':4,'ivec4':4,'uvec4':4}.get(typ,1)
            scalar=8 if typ=='uint64_t' else 2 if typ in ('uint16_t','float16_t') else 4
            size=scalar*components;align=scalar*(4 if components in (3,4) else 2 if components==2 else 1)
            for item in args(rest):
                v=re.match(r'\s*(\w+)(\s*\[\s*\])?',item)
                if not v:raise ValueError('unsupported RR buffer field: '+decl)
                if v[2]:result.append((v[1],HLSL_TYPE[typ],None,size))
                else:
                    off=(off+align-1)//align*align
                    result.append((v[1],HLSL_TYPE[typ],off,size));off+=size
        return result
    named=re.compile(r'layout\s*\(set\s*=\s*0,\s*binding\s*=\s*([^,)]+),\s*std430\)\s*(readonly\s+|writeonly\s+)?buffer\s+\w+\s*\{([^}]*)\}\s*(\w+)\s*;')
    pos=0
    while m:=named.search(source,pos):
        bind,ro,body,name=int_expr(m[1]),(m[2] or '').strip()=='readonly',m[3],m[4]
        # single arrays (weights, arena) keep the fast path below
        if re.fullmatch(r'\s*\w+\s+\w+\s*\[\s*\]\s*;\s*',body):pos=m.end();continue
        fields=struct_fields(body)
        resources.append(f'{"ByteAddressBuffer" if ro else "RWByteAddressBuffer"} rrB{bind} : register({"t" if ro else "u"}{bind});')
        source=source[:m.start()]+source[m.end():];pos=m.start()
        for field,typ,offset,size in fields:
            pattern=re.compile(r'\b'+re.escape(name)+r'\.'+re.escape(field)+r'\b')
            p=0
            while mm:=pattern.search(source,p):
                end=mm.end();j=end
                while j<len(source) and source[j].isspace():j+=1
                if offset is None:
                    if j>=len(source) or source[j]!='[':p=end;continue
                    stop=closing(source,j);index=source[j+1:stop];at=f'({index}) * {size}u'
                    after=source[stop+1:];store=re.match(r'\s*=\s*(?!=)',after)
                    if store:
                        if ro:raise ValueError('write to readonly '+name)
                        semi=source.index(';',stop+1);value=source[stop+1+store.end():semi]
                        repl=f'rrB{bind}.Store<{typ}>({at}, {value})';source=source[:mm.start()]+repl+source[semi:]
                    else:
                        repl=f'rrB{bind}.Load<{typ}>({at})';source=source[:mm.start()]+repl+source[stop+1:]
                    p=mm.start()+len(repl)
                else:
                    after=source[end:];store=re.match(r'\s*=\s*(?!=)',after)
                    if store:
                        if ro:raise ValueError('write to readonly '+name)
                        semi=source.index(';',end);value=source[end+store.end():semi]
                        repl=f'rrB{bind}.Store<{typ}>({offset}u, {value})';source=source[:mm.start()]+repl+source[semi:]
                    else:
                        repl=f'rrB{bind}.Load<{typ}>({offset}u)';source=source[:mm.start()]+repl+source[end:]
                    p=mm.start()+len(repl)
    def buffer(m):
        bind, readonly, body, name = int_expr(m[1]), (m[2] or '').strip()=='readonly', m[3], m[4]
        fields = re.fullmatch(r'\s*(\w+)\s+(\w+)\s*\[\s*\]\s*;\s*',body)
        if not fields: raise ValueError('unsupported RR buffer: '+body)
        typ, field=fields.groups(); buffers[name+'.'+field]=(bind,typ,readonly)
        return ''
    source = re.sub(r'layout\s*\(set\s*=\s*0,\s*binding\s*=\s*([^,)]+),\s*std430\)\s*(readonly\s+|writeonly\s+)?buffer\s+\w+\s*\{([^}]+)\}\s*(\w+)\s*;',buffer,source)
    for bind in sorted({b[0] for b in buffers.values()}):
        ro = all(v[2] for v in buffers.values() if v[0]==bind)
        resources.append(f'{"ByteAddressBuffer" if ro else "RWByteAddressBuffer"} rrB{bind} : register({"t" if ro else "u"}{bind});')
    for name,(bind,typ,ro) in buffers.items():
        pattern=re.compile(r'\b'+re.escape(name)+r'\s*\[')
        pos=0
        while m:=pattern.search(source,pos):
            start=source.index('[',m.start()); end=closing(source,start)
            index=source[start+1:end]; size={'uint':4,'int':4,'uint64_t':8,'float':4,'uint16_t':2,'float16_t':2,'uint8_t':1,'floate4m3_t':1,'vec2':8,'vec4':16,'uvec2':8,'uvec4':16}[typ]
            htyp={'uvec2':'uint2','uvec4':'uint4'}.get(typ,typ)  # the gfx11 stages read their weights as uvec4
            at=f'({index}) * {size}u'
            after=source[end+1:]
            store=re.match(r'\s*=\s*(?!=)',after)
            if store:
                if ro: raise ValueError('write to readonly '+name)
                stop=source.index(';',end+1); val=source[end+1+store.end():stop]
                replacement=f'rrB{bind}.Store<{htyp}>({at}, {val})'
                source=source[:m.start()]+replacement+source[stop:]
            else:
                if size==1:
                    read=f'((rrB{bind}.Load(({at}) & ~3u) >> ((({at}) & 3u) * 8u)) & 255u)'
                    if typ=='floate4m3_t':read='rrDecode('+read+')'
                else: read=f'rrB{bind}.Load<{htyp}>({at})'
                replacement=read; source=source[:m.start()]+read+source[end+1:]
            pos=m.start()+len(replacement)
    source=vector_loads(source)
    def texture(m):
        bind,name=int_expr(m[1]),m[2]; resources.append(f'D4R_SAMPLER({name}, t{bind}, s{bind})'); return ''
    source=re.sub(r'layout\s*\(set\s*=\s*0,\s*binding\s*=\s*([^,)]+)\)\s*uniform sampler2D\s+(\w+)\s*;',texture,source)
    def image(m):
        bind,fmt,name=int_expr(m[1]),m[2],m[3]; resources.append(f'RWTexture2D<float4> {name} : register(u{bind});'); return ''
    source=re.sub(r'layout\s*\(set\s*=\s*0,\s*binding\s*=\s*([^,)]+),\s*(\w+)\)\s*uniform writeonly image2D\s+(\w+)\s*;',image,source)
    source=re.sub(r'layout\(constant_id\s*=\s*0\)\s*const bool RR_ROBUST_IMAGES\s*=\s*false;', 'const bool RR_ROBUST_IMAGES = false;',source)
    source=re.sub(r'spirv_instruction\([^\n]+?\)\s+(\w+)\s+(\w+)\(([^;]+)\);', lambda m:f'{m[1]} {m[2]}({m[3]}) {{ return {"min" if "min" in m[2] else "max"}(a,b); }}',source)
    source=re.sub(r'(\d+(?:\.\d*)?(?:e[-+]?\d+)?)hf\b',r'\1h',source)
    source=re.sub(r'^uint16_t RR_ZERO16;', 'static uint16_t RR_ZERO16;',source,flags=re.M)
    # GLSL array constructors (including nested constructors) become initializers.
    while m:=re.search(r'\b\w+(?:\s*\[\s*\d*\s*\])+\s*\(',source):
        start=source.index('(',m.start()); end=closing(source,start)
        source=source[:m.start()]+'{'+source[start+1:end]+'}'+source[end+1:]
    lines,size=hlsl_build.translate(source)
    # HLSL globals are read-only; the model's mutable global state (the split
    # gather's rr2_xf, the hybrid tail's packed registers) needs `static`.
    globals_=re.compile(r'^((?:bool|int|uint|float|double|half|uint8_t|int8_t|uint16_t|int16_t|uint32_t|int32_t|'
                        r'uint64_t|int64_t|float16_t|float2|float3|float4|int2|int3|int4|uint2|uint3|uint4|'
                        r'bool2|bool3|bool4)\s+[^;{}]+;)$')
    lines=[globals_.sub(r'static \1',line) for line in lines]
    if not size: raise ValueError('missing workgroup size')
    threads=re.search(r'local_size_x\s*=\s*(\d+)',source)[1]
    init=(f'rrScratchBase = (gl_WorkGroupID.x + ((rrB0.Load(0) + (rrB0.Load(80) & 65535u) + 7u)/8u) * gl_WorkGroupID.y) * {spill_words}u;' if spill_words else '')
    lines=hoist_weight_loads('\n'.join(lines)).split('\n')
    groups=[]
    if param_groups:
        # The D3D12 runtime has no gl_NumWorkGroups; the F kernels use it to
        # reverse the window order (enc0's split) and to index phase grids.
        # The params' level width/height and spatial shift give the dispatch
        # grid ((lw + sx + 7) / 8, (lh + sy + 7) / 8, 1), as the Vulkan runtime;
        # a phase's x grid is that many times its part count.
        gx='(rrB0.Load(0u) + (rrB0.Load(80u) & 65535u) + 7u) / 8u'
        gy='(rrB0.Load(4u) + (rrB0.Load(80u) >> 16u) + 7u) / 8u'
        scale=gx if param_groups==1 else f'(({gx}) * {int(param_groups)}u)'
        # Once per invocation: a uniform value in registers, not three buffer
        # loads every time scBase()/the split code reads gl_NumWorkGroups.
        groups=['#undef gl_NumWorkGroups','#define gl_NumWorkGroups rrNumGroups_']
        resources.append('static uint3 rrNumGroups_;')
        init=f'rrNumGroups_ = uint3({scale}, {gy}, 1u); '+init
    result = '\n'.join(['#include "dialect_hlsl.h"',f'#define D4R_THREADS {threads}',f'#define RR_SPILL_WORDS {spill_words}',
        *(['#define RR_GFX11 1'] if gfx11 else []),
        '#include "rr_wmma.hlsli"',*groups,*resources,*lines,
        '#define D4R_HLSL_INIT_CALL '+init, '#define D4R_ENTRY_SIZE(...) D4R_ENTRY(__VA_ARGS__)', 'D4R_ENTRY_SIZE(D4R_WORKGROUP_SIZE)',''])
    if speed:
        result=re.sub(r'(\[loop\] for \(uint (?:hd|br) = 0u; (?:hd|br) < )([^;]+);',
                      lambda m:m[1]+'('+m[2]+f' + rrB0.Load({loop_zero_offset}u));',result)
        # gfx11 takes the gfx12 build's relaxations too (RX 7700 XT, RR D 1024x576 -> 2048x1152 replay: 32.1 -> 27.4 ms;
        # 61-67 dB against the exact Vulkan gfx11 route either way); D4R_RR_GFX11_EXACT=1 keeps the exact stages
        if not os.environ.get('D4R_TMP_STRICT') and not (gfx11 and os.environ.get('D4R_RR_GFX11_EXACT') == '1'): # TMP experiment
            result=relaxed_math(result,precise)
    return relaxed_math(result,precise) if relaxed and not speed else result


if __name__ == '__main__':
    Path(sys.argv[2]).write_text(translate(Path(sys.argv[1]).read_text()))
