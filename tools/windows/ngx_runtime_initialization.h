#pragma once
#include "ngx_windows_runtime.h"

namespace d4r::win {
// Own the complete Init contract when native RR defers the independent SR
// runtime. The caller may mutate or free its strings/path list after Init.
struct RuntimeInitialization {
    ComPtr<ID3D12Device> device;
    unsigned long long application;
    unsigned sdk;
    std::wstring data;
    ngx::FeatureCommonInfo common{};
    std::vector<std::wstring> paths;
    std::vector<const wchar_t*> pathPointers;
    ngx::ProjectIdentity project;
    bool hasCommon = false, hasProject = false;
    RuntimeInitialization(ID3D12Device* d3d, unsigned long long app, const wchar_t* directory, unsigned version,
        const ngx::FeatureCommonInfo* info, const ngx::ProjectIdentity* identity)
        : device(d3d), application(app), sdk(version), data(directory && *directory ? directory : std::filesystem::current_path().wstring()) {
        if (!d3d) throw std::runtime_error("Null NGX initialization device");
        if (identity) { project = *identity; hasProject = true; }
        if (info) {
            if (info->PathListInfo.Length > 1024 || (info->PathListInfo.Length && !info->PathListInfo.Path))
                throw std::runtime_error("Invalid NGX initialization path list");
            hasCommon = true;
            // Runtime consumes only path and logging information. InternalData
            // is opaque and must not become a deferred borrowed pointer.
            if (sdk >= 0x14) common.LoggingInfo = info->LoggingInfo;
            for (unsigned i = 0; i < info->PathListInfo.Length; ++i)
                if (info->PathListInfo.Path[i]) paths.emplace_back(info->PathListInfo.Path[i]);
            for (const auto& path : paths) pathPointers.push_back(path.c_str());
            common.PathListInfo = {pathPointers.data(), unsigned(pathPointers.size())};
        }
    }
    RuntimeInitialization(const RuntimeInitialization&) = delete;
    RuntimeInitialization& operator=(const RuntimeInitialization&) = delete;
    std::shared_ptr<Runtime> create() const {
        return std::make_shared<Runtime>(device.Get(), application, data.c_str(), sdk,
            hasCommon ? &common : nullptr, hasProject ? &project : nullptr);
    }
};
}
