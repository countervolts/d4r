// d4r_d3d12_split.h: the native D3D12 counterpart of vkd3d-proton's SplitCommandListForExternalWait
// (patches/vkd3d-proton 0001). NGX hands the shim the game's open command list in the middle of a frame; DLSS can only
// run once the inputs recorded so far have executed, and the game's later work in the same list reads the result.
// begin() splits the list at that point: whatever the game (or the shim) records into it afterwards goes into a
// second list d4r owns, and ExecuteCommandLists submits the first part, a queue Wait until `fence` reaches `value`
// (signalled once DLSS has written its result), then the second part. The game sees one list.
//
// Recording is redirected by replacing every recording method of ID3D12GraphicsCommandList (vtable slots from
// ClearState on) with a generic forwarding stub: it maps the list being called to the list it records into, keeps
// every argument register as it was and jumps to the original method, so no method needs its own wrapper. Close
// closes both parts; Reset of a part that was never executed drops the split (discarded). The second part starts
// with no bound state, as NGX's evaluate leaves the list's state undefined for the caller anyway.
//
// Include in one translation unit; needs d4r_d3d12_submit.h's hooks (installed first).
#pragma once
#include "d4r_d3d12_submit.h"
#include <atomic>
#include <cstdint>
#include <mutex>
#include <vector>

namespace d4r_split {
constexpr size_t CloseSlot = 9, FirstRedirected = 11, MaxSlots = 86, MaxSplits = 16;

struct Split {
    std::atomic<ID3D12GraphicsCommandList*> list{nullptr}; // the game's list
    std::atomic<bool> recording{false};                    // calls on `list` go to `second`
    ID3D12GraphicsCommandList* second = nullptr;
    ID3D12Fence* fence = nullptr;  // the second part waits until it reaches value
    uint64_t value = 0;
    ID3D12Fence* retire = nullptr; // signalled to retireValue on the same queue after the second part
    uint64_t retireValue = 0;
    bool closed = false;
    void (*done)(void* owner, bool executed) = nullptr;
    void* owner = nullptr;
    // Called on the thread that executes the list, after the first part has been submitted and before the second
    // part's wait: it may block (d3dhip runs the HIP network here) and queue work on the same queue. `fence` is
    // signalled to `value` from the CPU once it returns, so the wait cannot hang.
    void (*beforeSecond)(void* owner, ID3D12CommandQueue* queue) = nullptr;
    void* beforeOwner = nullptr;
};

namespace detail {
using Close = HRESULT(STDMETHODCALLTYPE*)(ID3D12GraphicsCommandList*);
inline Split splits[MaxSplits];
inline std::atomic<int> active{0};
inline std::mutex mutex;
inline Close close = nullptr;
inline std::atomic<bool> ready{false};
} // namespace detail
} // namespace d4r_split

extern "C" {
// the original methods, by vtable slot (used by the stubs below)
__attribute__((used)) void* d4r_split_original[d4r_split::MaxSlots] = {};
extern void* d4r_split_thunks[d4r_split::MaxSlots];

// The list a call on `self` records into: the second part while `self` is split, otherwise `self`.
__attribute__((used)) void* d4r_split_target(void* self)
{
    using namespace d4r_split::detail;
    if (active.load(std::memory_order_acquire) == 0)
        return self;
    for (auto& split : splits)
        if (split.list.load(std::memory_order_acquire) == self && split.recording.load(std::memory_order_acquire))
            return split.second;
    return self;
}
}

// One stub per redirected slot: the four argument registers and xmm0-3 survive the lookup, the stack arguments are
// untouched (the stub leaves its frame before the jump), and rcx becomes the list that records.
asm(R"(
    .text
    .altmacro
    .macro d4r_split_stub n
    .globl d4r_split_thunk_\n
d4r_split_thunk_\n:
    push %rcx
    push %rdx
    push %r8
    push %r9
    sub $0x68, %rsp
    movdqu %xmm0, 0x20(%rsp)
    movdqu %xmm1, 0x30(%rsp)
    movdqu %xmm2, 0x40(%rsp)
    movdqu %xmm3, 0x50(%rsp)
    call d4r_split_target
    movdqu 0x20(%rsp), %xmm0
    movdqu 0x30(%rsp), %xmm1
    movdqu 0x40(%rsp), %xmm2
    movdqu 0x50(%rsp), %xmm3
    add $0x68, %rsp
    pop %r9
    pop %r8
    pop %rdx
    pop %rcx
    mov %rax, %rcx
    jmp *(d4r_split_original + 8 * \n)(%rip)
    .endm
    .macro d4r_split_entry n
    .quad d4r_split_thunk_\n
    .endm
    .set d4r_split_i, 11
    .rept 75
    d4r_split_stub %d4r_split_i
    .set d4r_split_i, d4r_split_i + 1
    .endr
    .section .rdata,"dr"
    .p2align 3
    .globl d4r_split_thunks
d4r_split_thunks:
    .rept 11
    .quad 0
    .endr
    .set d4r_split_i, 11
    .rept 75
    d4r_split_entry %d4r_split_i
    .set d4r_split_i, d4r_split_i + 1
    .endr
    .noaltmacro
    .text
)");

namespace d4r_split {
namespace detail {
inline Split* find(ID3D12GraphicsCommandList* list)
{
    for (auto& split : splits)
        if (split.list.load(std::memory_order_acquire) == list)
            return &split;
    return nullptr;
}

// mutex held
inline void finish(Split& split, bool executed)
{
    split.recording.store(false, std::memory_order_release);
    split.list.store(nullptr, std::memory_order_release);
    active.fetch_sub(1, std::memory_order_acq_rel);
    if (split.done)
        split.done(split.owner, executed);
    split.done = nullptr;
    split.beforeSecond = nullptr;
    split.beforeOwner = nullptr;
}

inline HRESULT STDMETHODCALLTYPE close_hook(ID3D12GraphicsCommandList* list)
{
    HRESULT second = S_OK;
    {
        std::lock_guard<std::mutex> lock(mutex);
        if (Split* split = find(list); split != nullptr && split->recording.load())
        {
            split->recording.store(false, std::memory_order_release);
            second = close(split->second);
            split->closed = true;
        }
    }
    const HRESULT first = close(list);
    return FAILED(second) ? second : first;
}

// a split list reset before it was executed: its second part is dropped with it
inline void reset_hook(ID3D12GraphicsCommandList* list)
{
    std::lock_guard<std::mutex> lock(mutex);
    Split* split = find(list);
    if (split == nullptr)
        return;
    if (split->recording.load())
        close(split->second);
    finish(*split, false);
}

inline bool execute_hook(ID3D12CommandQueue* queue, UINT count, ID3D12CommandList* const* lists,
                         d4r_submit::detail::Execute execute)
{
    if (active.load(std::memory_order_acquire) == 0)
        return false;
    std::lock_guard<std::mutex> lock(mutex);
    std::vector<Split*> found(count, nullptr);
    bool any = false;
    for (UINT i = 0; i < count; ++i)
    {
        Split* split = find(static_cast<ID3D12GraphicsCommandList*>(lists[i]));
        if (split != nullptr && split->closed)
        {
            found[i] = split;
            any = true;
        }
    }
    if (!any)
        return false;
    std::vector<ID3D12CommandList*> chunk;
    std::vector<Split*> retiring;
    for (UINT i = 0; i < count; ++i)
    {
        chunk.push_back(lists[i]);
        if (Split* split = found[i])
        {
            execute(queue, static_cast<UINT>(chunk.size()), chunk.data());
            if (split->beforeSecond != nullptr)
            {
                split->beforeSecond(split->beforeOwner, queue);
                // the hook's work is done: satisfy the wait below from the CPU (a split without a hook is completed
                // by its owner's own signal, as before)
                if (split->fence != nullptr)
                    split->fence->Signal(split->value);
            }
            queue->Wait(split->fence, split->value);
            chunk.assign(1, split->second);
            retiring.push_back(split);
        }
    }
    execute(queue, static_cast<UINT>(chunk.size()), chunk.data());
    for (Split* split : retiring)
    {
        if (split->retire != nullptr)
            queue->Signal(split->retire, split->retireValue);
        finish(*split, true);
    }
    return true;
}
} // namespace detail

inline bool installed() { return detail::ready.load(); }

// Replaces the recording methods of the command lists `device` creates (once per process). Needs the submission
// hooks. False: lists cannot be split (the reason in *failure).
inline bool install(ID3D12Device* device, const char** failure = nullptr)
{
    using namespace detail;
    static std::mutex installMutex;
    std::lock_guard<std::mutex> lock(installMutex);
    if (ready.load())
        return true;
    if (!d4r_submit::install(device, failure))
        return false;
    ID3D12CommandAllocator* allocator = nullptr;
    ID3D12GraphicsCommandList* list = nullptr;
    if (FAILED(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&allocator))) ||
        FAILED(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, allocator, nullptr, IID_PPV_ARGS(&list))))
    {
        if (allocator) allocator->Release();
        if (failure) *failure = "the probe list could not be created";
        return false;
    }
    // the vtable's length is that of the newest interface the runtime implements; patching past it would
    // corrupt whatever follows, so a newer interface this header does not know stops at the last one it does
    static const GUID list10 = {0x7013c015, 0xd161, 0x4b63, {0xa0, 0x8c, 0x23, 0x85, 0x52, 0xdd, 0x8a, 0xcc}};
    static const GUID list9 = {0x34ed2808, 0xffe6, 0x4c2b, {0xb1, 0x1a, 0xca, 0xbd, 0x2b, 0x0c, 0x59, 0xe1}};
    static const GUID list8 = {0xee936ef9, 0x599d, 0x4d28, {0x93, 0x8e, 0x23, 0xc4, 0xad, 0x05, 0xce, 0x51}};
    const struct { const GUID* iid; size_t slots; } versions[] = {
        {&list10, 86}, {&list9, 84}, {&list8, 82}, {&__uuidof(ID3D12GraphicsCommandList7), 81},
        {&__uuidof(ID3D12GraphicsCommandList6), 80}, {&__uuidof(ID3D12GraphicsCommandList5), 79},
        {&__uuidof(ID3D12GraphicsCommandList4), 77}, {&__uuidof(ID3D12GraphicsCommandList3), 68},
        {&__uuidof(ID3D12GraphicsCommandList2), 67}, {&__uuidof(ID3D12GraphicsCommandList1), 66}};
    size_t slots = 60;
    for (const auto& version : versions)
    {
        IUnknown* newer = nullptr;
        if (SUCCEEDED(list->QueryInterface(*version.iid, reinterpret_cast<void**>(&newer))))
        {
            newer->Release();
            slots = version.slots;
            break;
        }
    }
    void** vtable = *reinterpret_cast<void***>(list);
    bool ok = true;
    void* closeOriginal = nullptr;
    ok = d4r_submit::detail::patch(vtable, CloseSlot, reinterpret_cast<void*>(&close_hook), &closeOriginal);
    if (closeOriginal) close = reinterpret_cast<Close>(closeOriginal);
    for (size_t slot = FirstRedirected; ok && slot < slots; ++slot)
    {
        void* original = nullptr;
        ok = d4r_submit::detail::patch(vtable, slot, d4r_split_thunks[slot], &original);
        if (original) d4r_split_original[slot] = original;
        ok = ok && d4r_split_original[slot] != nullptr;
    }
    list->Close();
    list->Release();
    allocator->Release();
    if (!ok || close == nullptr)
    {
        if (failure) *failure = "the command list vtable could not be patched";
        return false;
    }
    auto& s = d4r_submit::detail::state();
    s.splitExecute = &execute_hook;
    s.splitReset = &reset_hook;
    ready = true;
    return true;
}

// From now on what is recorded into `list` goes into `second` (open, of the same type). When `list` is executed,
// `second` follows it on the same queue once `fence` reaches `value`, and `retire` is then signalled to
// `retireValue` behind it. `done(owner, executed)` reports the outcome once: executed, or discarded (reset before it
// was executed). `beforeSecond(beforeOwner, queue)` runs between the two parts (see Split). False: no split slot was
// free (nothing changes).
inline bool begin(ID3D12GraphicsCommandList* list, ID3D12GraphicsCommandList* second, ID3D12Fence* fence,
                  uint64_t value, ID3D12Fence* retire, uint64_t retireValue, void (*done)(void*, bool), void* owner,
                  void (*beforeSecond)(void*, ID3D12CommandQueue*) = nullptr, void* beforeOwner = nullptr)
{
    using namespace detail;
    if (!ready.load())
        return false;
    std::lock_guard<std::mutex> lock(mutex);
    if (find(list) != nullptr)
        return false;
    for (auto& split : splits)
    {
        if (split.list.load() != nullptr)
            continue;
        split.second = second;
        split.fence = fence;
        split.value = value;
        split.retire = retire;
        split.retireValue = retireValue;
        split.closed = false;
        split.done = done;
        split.owner = owner;
        split.beforeSecond = beforeSecond;
        split.beforeOwner = beforeOwner;
        split.recording.store(true, std::memory_order_release);
        split.list.store(list, std::memory_order_release);
        active.fetch_add(1, std::memory_order_acq_rel);
        return true;
    }
    return false;
}
} // namespace d4r_split
