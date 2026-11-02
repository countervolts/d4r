"""Conversion labels must belong to the older Vulkan query slot, not this frame."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]


class VramProfileTimingTests(unittest.TestCase):
    def test_older_slot_mode_labels(self):
        source=(ROOT/'tools/d4r_nvngx_shim.cpp').read_text()
        start=source.index('static void read_profile_timestamps(Feature& feature, uint32_t frame, FrameTiming& timing)\n{')
        end=source.index('\n}\n',start)+3
        program=r'''
#include <cassert>
#include <cstdint>
constexpr unsigned kOutputSlots=4;
struct FrameTiming {double vkInput=-1,vkDlssWait=-1,vkOutput=-1,vkTotal=-1;int vkOutputFused=-1,vkInputFused=-1;};
struct Feature {uint64_t values[16]={};volatile uint64_t* timestampValues=values;double timestampTicksPerMs=10;
 uint8_t timestampOutputFused[4]={},timestampInputFused[4]={};bool recordOutputFused=true,recordInputFused=true;};
''' + source[start:end] + r'''
int main() {
 Feature f;FrameTiming old;
 f.values[0]=10;f.values[1]=20;f.values[2]=50;f.values[3]=60;
 f.timestampOutputFused[0]=0;f.timestampInputFused[0]=1;
 read_profile_timestamps(f,300,old); // frame 296 used transfer output, despite current mode switching on
 assert(old.vkOutputFused==0 && old.vkInputFused==1 && old.vkTotal==5);
 assert(!f.recordOutputFused && !f.recordInputFused && f.values[0]==0);
 f.values[0]=70;f.values[1]=80;f.values[2]=110;f.values[3]=120;
 f.timestampOutputFused[0]=1;f.timestampInputFused[0]=0;FrameTiming next;
 read_profile_timestamps(f,304,next);
 assert(next.vkOutputFused==1 && next.vkInputFused==0);
 FrameTiming unavailable;read_profile_timestamps(f,308,unavailable);
 assert(unavailable.vkOutputFused==-1 && unavailable.vkInputFused==-1 && unavailable.vkTotal==-1);
}
'''
        with tempfile.TemporaryDirectory() as temp:
            runner=Path(temp)/'timing'
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-x','c++','-',
                            '-o',str(runner)],input=program,text=True,check=True,capture_output=True)
            subprocess.run([str(runner)],check=True,timeout=5)
