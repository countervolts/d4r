@ stdcall cuInit(long) cuInit
@ stdcall cuDeviceGetCount(ptr) cuDeviceGetCount
@ stdcall cuDeviceGet(ptr long) cuDeviceGet
@ stdcall cuCtxPushCurrent_v2(ptr) cuCtxPushCurrent_v2
@ stdcall cuCtxPopCurrent_v2(ptr) cuCtxPopCurrent_v2
@ stdcall cuCtxGetDevice(ptr) cuCtxGetDevice
@ stdcall cuCtxCreate_v2(ptr long long) cuCtxCreate_v2
@ stdcall cuCtxDestroy_v2(ptr) cuCtxDestroy_v2
@ stdcall cuCtxSetCurrent(ptr) cuCtxSetCurrent
@ stdcall cuCtxSynchronize() cuCtxSynchronize
@ stdcall cuDeviceGetLuid(ptr ptr long) cuDeviceGetLuid
@ stdcall cuDeviceGetAttribute(ptr long long) cuDeviceGetAttribute
@ stdcall cuDeviceGetUuid(ptr long) cuDeviceGetUuid
@ stdcall cuGetErrorString(long ptr) cuGetErrorString
@ stdcall cuGetProcAddress(ptr ptr long long) cuGetProcAddress
@ stdcall cuGetProcAddress_v2(ptr ptr long long ptr) cuGetProcAddress_v2
@ stdcall cuArrayCreate(ptr ptr) cuArrayCreate
@ stdcall cuArray3DCreate(ptr ptr) cuArray3DCreate
@ stdcall cuArrayDestroy(ptr) cuArrayDestroy
@ stdcall cuArrayGetDescriptor(ptr ptr) cuArrayGetDescriptor
@ stdcall cuDestroyExternalMemory(ptr) cuDestroyExternalMemory
@ stdcall cuLaunchKernel(ptr long long long long long long long ptr ptr ptr) cuLaunchKernel
@ stdcall cuMemAlloc(ptr long long) cuMemAlloc
@ stdcall cuMemAllocHost(ptr long long) cuMemAllocHost
@ stdcall cuMemFree(long long) cuMemFree
@ stdcall cuMemFreeHost(ptr) cuMemFreeHost
@ stdcall cuMemcpy2D(ptr) cuMemcpy2D
@ stdcall cuMemcpyDtoH(ptr long long) cuMemcpyDtoH
@ stdcall cuMemcpyHtoDAsync(long long ptr long long ptr) cuMemcpyHtoDAsync
@ stdcall cuMipmappedArrayDestroy(ptr) cuMipmappedArrayDestroy
@ stdcall cuModuleGetFunction(ptr ptr ptr) cuModuleGetFunction
@ stdcall cuModuleLoad(ptr ptr) cuModuleLoad
@ stdcall cuModuleLoadData(ptr ptr) cuModuleLoadData
@ stdcall cuModuleLoadDataEx(ptr ptr long ptr ptr) cuModuleLoadDataEx
@ stdcall cuModuleUnload(ptr) cuModuleUnload
@ stdcall cuSurfObjectCreate(ptr ptr) cuSurfObjectCreate
@ stdcall cuSurfObjectDestroy(long long) cuSurfObjectDestroy
@ stdcall cuSurfObjectGetResourceDesc(ptr long long) cuSurfObjectGetResourceDesc
@ stdcall cuTexObjectCreate(ptr ptr ptr ptr) cuTexObjectCreate
@ stdcall cuTexObjectDestroy(long long) cuTexObjectDestroy
@ stdcall cuTexObjectGetResourceDesc(ptr long long) cuTexObjectGetResourceDesc
@ stdcall cuEventCreate(ptr ptr) cuEventCreate
@ stdcall cuEventDestroy(ptr) cuEventDestroy
@ stdcall cuEventDestroy_v2(ptr) cuEventDestroy_v2
@ stdcall cuEventRecord(ptr ptr) cuEventRecord
@ stdcall cuEventRecordWithFlags(ptr ptr ptr) cuEventRecordWithFlags
@ stdcall cuEventSynchronize(ptr) cuEventSynchronize
@ stdcall cuEventQuery(ptr) cuEventQuery
@ stdcall cuEventElapsedTime(ptr ptr ptr) cuEventElapsedTime
@ stdcall cuStreamCreate(ptr ptr) cuStreamCreate
@ stdcall cuStreamCreateWithPriority(ptr ptr ptr) cuStreamCreateWithPriority
@ stdcall cuStreamDestroy(ptr) cuStreamDestroy
@ stdcall cuStreamDestroy_v2(ptr) cuStreamDestroy_v2
@ stdcall cuStreamSynchronize(ptr) cuStreamSynchronize
@ stdcall cuStreamQuery(ptr) cuStreamQuery
@ stdcall cuStreamWaitEvent(ptr ptr ptr) cuStreamWaitEvent
@ stdcall cuStreamGetFlags(ptr ptr) cuStreamGetFlags
@ stdcall cuStreamGetPriority(ptr ptr) cuStreamGetPriority
@ stdcall cuCtxGetCurrent(ptr) cuCtxGetCurrent
@ stdcall cuCtxGetStreamPriorityRange(ptr ptr) cuCtxGetStreamPriorityRange
@ stdcall cuCtxGetLimit(ptr ptr) cuCtxGetLimit
@ stdcall cuCtxSetLimit(ptr ptr) cuCtxSetLimit
@ stdcall cuDriverGetVersion(ptr) cuDriverGetVersion
@ stdcall cuDeviceGetName(ptr ptr ptr) cuDeviceGetName
@ stdcall cuDeviceTotalMem(ptr ptr) cuDeviceTotalMem
@ stdcall cuDeviceTotalMem_v2(ptr ptr) cuDeviceTotalMem_v2
@ stdcall cuDeviceComputeCapability(ptr ptr ptr) cuDeviceComputeCapability
@ stdcall cuMemGetInfo(ptr ptr) cuMemGetInfo
@ stdcall cuMemGetInfo_v2(ptr ptr) cuMemGetInfo_v2
@ stdcall cuMemcpyHtoD(ptr ptr ptr) cuMemcpyHtoD
@ stdcall cuMemcpyHtoD_v2(ptr ptr ptr) cuMemcpyHtoD_v2
@ stdcall cuMemcpyDtoD(ptr ptr ptr) cuMemcpyDtoD
@ stdcall cuMemcpyDtoDAsync(ptr ptr ptr ptr) cuMemcpyDtoDAsync
@ stdcall cuMemcpyDtoHAsync(ptr ptr ptr ptr) cuMemcpyDtoHAsync
@ stdcall cuMemsetD8(ptr ptr ptr) cuMemsetD8
@ stdcall cuMemsetD16(ptr ptr ptr) cuMemsetD16
@ stdcall cuMemsetD32(ptr ptr ptr) cuMemsetD32
@ stdcall cuMemsetD8Async(ptr ptr ptr ptr) cuMemsetD8Async
@ stdcall cuMemsetD16Async(ptr ptr ptr ptr) cuMemsetD16Async
@ stdcall cuMemsetD32Async(ptr ptr ptr ptr) cuMemsetD32Async
@ stdcall cuMemsetD2D8(ptr ptr ptr ptr ptr) cuMemsetD2D8
@ stdcall cuMemsetD2D32(ptr ptr ptr ptr ptr) cuMemsetD2D32
@ stdcall cuFuncSetAttribute(ptr ptr ptr) cuFuncSetAttribute
@ stdcall cuFuncGetAttribute(ptr ptr ptr) cuFuncGetAttribute
@ stdcall cuModuleGetGlobal(ptr ptr ptr ptr) cuModuleGetGlobal
@ stdcall cuOccupancyMaxActiveBlocksPerMultiprocessor(ptr ptr ptr ptr) cuOccupancyMaxActiveBlocksPerMultiprocessor
@ stdcall cuOccupancyMaxPotentialBlockSize(ptr ptr ptr ptr ptr ptr) cuOccupancyMaxPotentialBlockSize
@ stdcall cuMemAllocPitch(ptr ptr ptr ptr ptr) cuMemAllocPitch
@ stdcall cuMemHostAlloc(ptr ptr ptr) cuMemHostAlloc
@ stdcall cuPointerGetAttribute(ptr ptr ptr) cuPointerGetAttribute
@ stdcall d4rImportVulkanMemory(ptr int64 int64 ptr ptr) d4rImportVulkanMemory
@ stdcall d4rReleaseVulkanMemory(ptr) d4rReleaseVulkanMemory
@ stdcall d4rMemcpy2DAsync(ptr ptr) d4rMemcpy2DAsync
@ stdcall d4rCtxSynchronize() d4rCtxSynchronize
@ stdcall d4rEventSynchronize(ptr) d4rEventSynchronize
@ stdcall d4rStreamWaitValue32(int64 long) d4rStreamWaitValue32
@ stdcall d4rWriteValue32(int64 long) d4rWriteValue32
@ stdcall d4rSetArrayRedirect(ptr int64 long) d4rSetArrayRedirect
@ stdcall d4rRegisterLinearTexture(int64 int64 int64 long long) d4rRegisterLinearTexture
@ stdcall d4rSetEnv(str str long) d4rSetEnv
@ stdcall d4rLoadError() d4rLoadError
@ stdcall d4rOutputKernelNative() d4rOutputKernelNative
