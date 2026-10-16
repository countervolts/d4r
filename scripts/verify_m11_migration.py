#!/usr/bin/env python3
"""Verify all original M11 source pack bytes remain in this reorganized tree."""
import csv, hashlib
from pathlib import Path
root=Path(__file__).resolve().parent.parent
manifest=root/'docs/archive/m11-source-snapshot/MIGRATION_MAP.tsv'
bad=[]
with manifest.open(newline='',encoding='utf-8') as f:
 for row in csv.DictReader(f,delimiter='\t'):
  path=root/row['new_path']
  if not path.is_file():
   bad.append('MISSING '+row['new_path']);continue
  sha=hashlib.sha256(path.read_bytes()).hexdigest()
  if sha!=row['sha256'] or path.stat().st_size!=int(row['bytes']):
   bad.append('MISMATCH '+row['new_path'])
if bad:
 print('\n'.join(bad[:50]));raise SystemExit(f'FAILED: {len(bad)} mismatches')
print('OK: source file bytes and paths match MIGRATION_MAP.tsv')
