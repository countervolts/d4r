#!/usr/bin/env python3
"""A copy of a compiled preset L model with only its input and expansion stages (enc0_l, dec0_l, both plane layouts)
rebuilt from the working tree's engine/layer_m.comp, with the relaxed build's defines replaced by LVAR_BASE (default
'-DM_RELAXED -DWPW=2 -DM_RESIDUAL_F32'; empty for the exact build) plus DEFINES. model.json is rehashed.

Used for the experiments in docs/amd-vector-issue-limit.md: count a variant's instructions (tools/radv_isa_mix.py),
time it (d4r-m D4R_M_TIME_EACH=1) and compare its output, without recompiling the model from its capture.

usage: build_l_stage_variant.py SRC_MODEL DST_MODEL [DEFINE ...]
"""
import hashlib, json, os, shutil, subprocess, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, 'engine'))
import l_model  # noqa: E402,F401  (registers enc0l, dec0l)
from test_m import defines  # noqa: E402

src, dst, extra = sys.argv[1], sys.argv[2], sys.argv[3:]
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(src, dst)
base = os.environ.get('LVAR_BASE', '-DM_RELAXED -DWPW=2 -DM_RESIDUAL_F32').split()
for planes, suffix in ((0, ''), (1, '_planes')):
    for stage, name in (('enc0l', 'enc0_l'), ('dec0l', 'dec0_l')):
        subprocess.run(['glslangValidator', '-V', '--target-env', 'vulkan1.3', *defines(stage), f'-DPLANES={planes}', *base, *extra,
                        os.path.join(ROOT, 'engine', 'layer_m.comp'), '-o', f'{dst}/{name}{suffix}.spv'], check=True, stdout=subprocess.DEVNULL)
manifest = json.load(open(f'{dst}/model.json'))
manifest['files'] = {f: hashlib.sha256(open(f'{dst}/{f}', 'rb').read()).hexdigest() for f in manifest['files']}
json.dump(manifest, open(f'{dst}/model.json', 'w'), indent=1)
