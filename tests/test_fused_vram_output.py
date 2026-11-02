"""Exercise the actual optional output recorder's lifetime and fallback paths."""
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class FusedVramOutputTests(unittest.TestCase):
    def test_output_lifetime_and_fallback(self):
        shim=(ROOT/'tools/d4r_nvngx_shim.cpp').read_text()
        start=shim.index('static bool record_feature_vram_output(')
        end=shim.index('// --- Split frames',start)
        input_start=shim.index('static bool ensure_fused_color(')
        input_end=shim.index('// Records the input copies',input_start)
        program=r'''
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>
using VkImage=uint64_t;using VkImageView=uint64_t;using VkBuffer=uint64_t;
using VkCommandBuffer=uint64_t;using D3D12_RESOURCE_STATES=int;
constexpr int VK_SUCCESS=0,VK_NULL_HANDLE=0,VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO=1,
 VK_STRUCTURE_TYPE_MEMORY_BARRIER=2,VK_IMAGE_VIEW_TYPE_2D=3,VK_FORMAT_B10G11R11_UFLOAT_PACK32=4,
 VK_IMAGE_ASPECT_COLOR_BIT=1,VK_ACCESS_MEMORY_WRITE_BIT=1,VK_ACCESS_SHADER_READ_BIT=2,
 VK_ACCESS_SHADER_WRITE_BIT=4,VK_ACCESS_MEMORY_READ_BIT=8,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT=1,
 VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT=2,DXGI_FORMAT_R11G11B10_FLOAT=26,
 D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS=4,D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE=8,
 DXGI_FORMAT_R8G8B8A8_UNORM=28,VK_FORMAT_R8G8B8A8_UNORM=5,D3D12_RESOURCE_STATE_UNORDERED_ACCESS=8,
 D3D12_RESOURCE_STATE_COPY_DEST=16;
#define SUCCEEDED(value) ((value)>=0)
struct Range {int aspect,baseMip,levels,baseLayer,layers;};
struct VkImageViewCreateInfo {int sType;VkImage image=0;int viewType=0,format=0;Range subresourceRange={};};
struct VkMemoryBarrier {int sType;void* next;int src,dst;};
struct Desc {int Format=26,Flags=4;};
struct ID3D12Resource {Desc desc;int refs=0;Desc GetDesc(){return desc;}void AddRef(){++refs;}};
struct ID3D12GraphicsCommandList {};
struct Set {int pool=0;};
struct FusedColorInput {VkImage image=0;VkImageView view=0;VkBuffer buffer=0;Set set;ID3D12Resource* owner=nullptr;uint32_t inputMode=5;};
struct VramBuffer {VkBuffer buffer=1;size_t bytes=800;};
struct VramCopy {bool convert=true;unsigned width=10,height=10;VkImage image=10;};
struct Feature {bool fusedOutputAllowed=true,recordOutputFused=false,fusedColorAllowed=true;
 struct {VramBuffer vram[1];struct {size_t size(){return 800;}} host[1];} inputs[3];
 FusedColorInput fusedColor[3];std::vector<FusedColorInput> retiredFusedColor;struct {VramBuffer vram;} outputs[4];FusedColorInput fusedOutput[4];std::vector<FusedColorInput> retiredFusedOutput;int outputConversion=0;};
static unsigned enabled=1;
static bool g_fusedColorReady=true,viewOk=true,setOk=true,beginOk=true,endOk=true;
static int views=0,destroyed=0,frees=0,records=0,legacy=0;
static unsigned env_uint(const char*,unsigned){return enabled;}
static void logf(const char*,...) {}
static std::vector<int> transitions;
static void transition(ID3D12GraphicsCommandList*,ID3D12Resource*,int,int after){transitions.push_back(after);}
static bool record_vram_output(ID3D12GraphicsCommandList*,const VramBuffer&,const VramCopy&,int){++legacy;return true;}
struct Interop {int BeginVkCommandBufferInterop(ID3D12GraphicsCommandList*,VkCommandBuffer* c){*c=1;return beginOk?0:-1;}int EndVkCommandBufferInterop(ID3D12GraphicsCommandList*){return endOk?0:-1;}} interop;
static struct {
 int device=0;Interop* interop=&::interop;
 int createView(int,const VkImageViewCreateInfo*,void*,VkImageView* out){++views;if(viewOk)*out=5;return viewOk?0:-1;}
 void destroyView(int,VkImageView,void*){++destroyed;}
 void barrier(VkCommandBuffer,int,int,int,int,const VkMemoryBarrier*,int,void*,int,void*){}
} g_vk;
static struct {
 bool set(Set& s,VkImageView,int,VkBuffer,size_t,bool){s.pool=1;return setOk;}
 void free(Set& s){++frees;s={};}
 void record(VkCommandBuffer,const Set&,unsigned,unsigned){++records;}
} g_fusedColorStage;
''' + shim[input_start:input_end] + shim[start:end] + r'''
int main() {
 ID3D12GraphicsCommandList list;ID3D12Resource output,changed;VramCopy copy;Feature f;
 enabled=false;assert(record_feature_vram_output(f,0,&list,&output,8,copy));assert(legacy==1 && views==0 && records==0);
 enabled=true;assert(record_feature_vram_output(f,0,&list,&output,8,copy));assert(output.refs==1 && views==1 && records==1 && legacy==1);
 assert(record_feature_vram_output(f,0,&list,&output,8,copy));assert(output.refs==1 && views==1 && records==2);
 assert(record_feature_vram_output(f,0,&list,&changed,8,copy));assert(changed.refs==1 && output.refs==1 && f.retiredFusedOutput.size()==1);
 Feature missingUav;ID3D12Resource noUav;noUav.desc.Flags=0;int old=legacy;
 assert(record_feature_vram_output(missingUav,0,&list,&noUav,8,copy));assert(legacy==old+1 && !noUav.refs);
 Feature small;small.outputs[0].vram.bytes=799;old=legacy;
 assert(record_feature_vram_output(small,0,&list,&output,8,copy));assert(legacy==old+1);
 Feature packed;copy.convert=false;old=legacy;
 assert(record_feature_vram_output(packed,0,&list,&output,8,copy));assert(legacy==old+1);copy.convert=true;
 Feature noView;viewOk=false;old=legacy;
 assert(record_feature_vram_output(noView,0,&list,&output,8,copy));assert(legacy==old+1 && !noView.fusedOutputAllowed);viewOk=true;
 Feature noSet;setOk=false;int released=destroyed;old=legacy;
 assert(record_feature_vram_output(noSet,0,&list,&output,8,copy));assert(legacy==old+1 && destroyed==released+1 && !noSet.fusedOutputAllowed);setOk=true;
 Feature noBegin;beginOk=false;old=legacy;int oldRecords=records;
 assert(record_feature_vram_output(noBegin,0,&list,&output,8,copy));assert(legacy==old+1 && records==oldRecords && !noBegin.fusedOutputAllowed);beginOk=true;
 Feature noEnd;endOk=false;old=legacy;
 assert(record_feature_vram_output(noEnd,0,&list,&output,8,copy));assert(legacy==old+1 && !noEnd.fusedOutputAllowed);endOk=true;
 Feature bounded;bounded.retiredFusedOutput.resize(64);old=legacy;int oldViews=views;
 assert(record_feature_vram_output(bounded,0,&list,&output,8,copy));assert(legacy==old+1 && views==oldViews && !bounded.fusedOutputAllowed);
 enabled=2;Feature alternating;old=legacy;
 assert(record_feature_vram_output(alternating,0,&list,&output,8,copy,0));assert(legacy==old+1 && !alternating.recordOutputFused);
 assert(record_feature_vram_output(alternating,0,&list,&output,8,copy,300));assert(legacy==old+1 && alternating.recordOutputFused);
 assert(record_feature_vram_output(alternating,0,&list,&output,8,copy,600));assert(legacy==old+2 && !alternating.recordOutputFused);
 enabled=1;Feature input;ID3D12Resource r11;
 assert(ensure_fused_color(input,0,&r11,copy,0));assert(input.fusedColor[0].inputMode==1 && r11.refs==1);
 int inputViews=views;assert(ensure_fused_color(input,0,&r11,copy,0));assert(views==inputViews && r11.refs==1);
 enabled=2;assert(!ensure_fused_color(input,0,&r11,copy,300));assert(ensure_fused_color(input,0,&r11,copy,600));
 assert(!ensure_fused_color(input,0,&r11,copy,1200));assert(views==inputViews);
 enabled=1;ID3D12Resource rgba8;rgba8.desc.Format=28;Feature rgbaInput;
 assert(ensure_fused_color(rgbaInput,0,&rgba8,copy,0));assert(rgbaInput.fusedColor[0].inputMode==5);
 ID3D12Resource denied;denied.desc.Flags=8;assert(!ensure_fused_color(input,0,&denied,copy));
 Feature limitedInput;limitedInput.retiredFusedColor.resize(64);
 assert(!ensure_fused_color(limitedInput,0,&r11,copy) && !limitedInput.fusedColorAllowed);
 assert(transitions.back()==8); // every route restores caller's output state
}
'''
        with tempfile.TemporaryDirectory() as temp:
            runner=Path(temp)/'fused-output'
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-Wno-missing-field-initializers',
                            '-x','c++','-','-o',str(runner)],input=program,text=True,check=True,capture_output=True)
            subprocess.run([str(runner)],check=True,timeout=5)
