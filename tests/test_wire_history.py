"""Sparse wire publication rejects aliased scratch, inputs and overflowing ranges."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class WireHistoryTests(unittest.TestCase):
    def test_sparse_buffer_contract(self):
        source = r'''
#include "d4r_wire_history.h"
#include <assert.h>
#include <string.h>
int main(void) {
    D4rWireHistorySparseParams p; memset(&p,0,sizeof(p));
    D4rWireHistoryParams* b=&p.base;
    b->width=65;b->height=67;b->render_width=21;b->render_height=22;
    b->current=0x10000000;b->color=0x20000000;b->native_depth=0x30000000;
    b->motion=0x40000000;b->previous=0x50000000;b->previous_age=0x60000000;
    b->previous_depth=0x70000000;b->output=0x80000000;b->output_age=0x90000000;
    b->output_depth=0xa0000000;b->input_depth=0xb0000000;b->output_history=0xc0000000;
    b->color_pitch=21*8;b->depth_pitch=b->motion_pitch=65*4;
    b->output_pitch=65*8;b->input_depth_pitch=21*4;p.patch_mask=0xd0000000;
    assert(sizeof(p)==176 && d4r_wire_history_sparse_parameters_valid(&p));
    assert(!d4r_wire_history_sparse_parameters_valid(0));
    D4rWireHistorySparseParams q=p;q.patch_mask=0;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.patch_mask=UINT64_MAX-16;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.patch_mask=q.base.current+8;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.patch_mask=q.base.previous_age+1;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.base.current=q.base.color;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.base.output=q.base.native_depth+4;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.base.current=q.base.output_history+8;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.base.output_pitch=UINT64_MAX-7;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.base.current=UINT64_MAX-1024;assert(!d4r_wire_history_sparse_parameters_valid(&q));
    q=p;q.patch_mask=q.base.current+66*q.base.output_pitch+65*8;
    assert(d4r_wire_history_sparse_parameters_valid(&q));
    D4rWireHistoryPreparedParams prepared={p,0xe0000000};
    assert(sizeof(prepared)==184 && d4r_wire_history_prepared_parameters_valid(&prepared));
    D4rWireHistoryPreparedParams no=prepared;no.axes=0;assert(!d4r_wire_history_prepared_parameters_valid(&no));
    no=prepared;no.axes=no.sparse.base.current;assert(!d4r_wire_history_prepared_parameters_valid(&no));
    no=prepared;no.axes=no.sparse.patch_mask;assert(!d4r_wire_history_prepared_parameters_valid(&no));
    no=prepared;no.axes=UINT64_MAX-16;assert(!d4r_wire_history_prepared_parameters_valid(&no));
    return 0;
}
'''
        with tempfile.TemporaryDirectory() as tmp:
            runner = Path(tmp)/'wire-contract'
            for compiler, language, standard in (('cc', 'c', 'c11'), ('c++', 'c++', 'c++17')):
                subprocess.run([compiler, '-x', language, '-std='+standard, '-Wall', '-Wextra', '-Werror',
                                '-I', str(ROOT/'tools'), '-o', str(runner), '-'],
                               input=source, text=True, check=True, capture_output=True)
                subprocess.run([str(runner)], check=True)
