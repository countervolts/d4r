"""Evaluate the preprocessed integer pool extents (no identifiers or calls)."""
import re


def extent(s):
    s=re.sub(r'(\d+)u\b',r'\1',s).strip()
    # Strip balanced outer parentheses, then evaluate the one ternary generated
    # by swin_m.comp's max(12*CX,24*C) expression.
    def value(text):
        text=text.strip()
        depth=0
        for i,c in enumerate(text):
            if c=='(':depth+=1
            elif c==')':depth-=1
            elif c=='?' and depth==0:
                yes,no=text[i+1:].split(':',1)
                return value(yes) if value(text[:i]) else value(no)
        if text.startswith('('):
            depth=0
            for i,c in enumerate(text):
                if c=='(':depth+=1
                elif c==')':depth-=1
                if depth==0:break
            if i==len(text)-1:return value(text[1:-1])
        # Replace innermost parentheses (including ternaries) until ordinary
        # integer arithmetic remains.
        while '(' in text:
            text=re.sub(r'\([^()]*\)',lambda m:str(value(m[0][1:-1])),text)
        if not re.fullmatch(r'[\d\s+*/<>-]+',text):raise ValueError('invalid pool extent')
        return int(eval(text.replace('/','//'),{'__builtins__':{}},{}))
    return value(s)
