#pragma once
// Copyright (c) Microsoft Corporation.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.
// ABI subset of Microsoft's MIT-licensed DirectX-Headers, commit adbd6f3b.
// No pointers to CPU-owned nested structures exist in SetProgram's descriptor.
// The generic/raytracing variants use only the ProgramIdentifier field.
namespace d4r::win::commands {
struct NewWorkGraphDescription {
    UINT64 program_identifier[4];
    UINT flags;
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE backing_memory;
    D3D12_GPU_VIRTUAL_ADDRESS_RANGE_AND_STRIDE root_arguments;
};
struct NewProgramDescription {
    UINT type;
    union {
        UINT64 generic_program_identifier[4];
        UINT64 raytracing_program_identifier[4];
        NewWorkGraphDescription work_graph;
    };
};
static_assert(sizeof(NewWorkGraphDescription) == 80 && sizeof(NewProgramDescription) == 88,
              "Public Windows x64 SetProgram ABI");
inline constexpr GUID command_list8_iid = {0xee936ef9,0x599d,0x4d28,{0x93,0x8e,0x23,0xc4,0xad,0x05,0xce,0x51}};
inline constexpr GUID command_list9_iid = {0x34ed2808,0xffe6,0x4c2b,{0xb1,0x1a,0xca,0xbd,0x2b,0x0c,0x59,0xe1}};
inline constexpr GUID command_list10_iid = {0x7013c015,0xd161,0x4b63,{0xa0,0x8c,0x23,0x85,0x52,0xdd,0x8a,0xcc}};
}
