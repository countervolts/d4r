"""Canonical Windows build target registry, shared with CMake and PowerShell."""
import json
import pathlib

data = json.loads(pathlib.Path(__file__).with_name('gpu-targets.json').read_text())
if data['schema'] != 1:
    raise RuntimeError('Unsupported GPU target registry schema')
TARGETS = {row['architecture']: row for row in data['targets']}
if len(TARGETS) != len(data['targets']):
    raise RuntimeError('Duplicate GPU target')
