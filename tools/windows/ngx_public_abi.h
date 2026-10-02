#pragma once
#include <cstdint>
#include <string>

// Public NGX driver export ABI, also used by the upstream shim and OptiScaler.
// NVIDIA SDK headers or binaries are not build dependencies.
namespace d4r::ngx {
struct PathListInfo { const wchar_t* const* Path; unsigned Length; };
struct LoggingInfo { void* LoggingCallback; int MinimumLoggingLevel; bool DisableOtherLoggingSinks; };
struct FeatureCommonInfo { PathListInfo PathListInfo; void* InternalData; LoggingInfo LoggingInfo; };
struct ProjectDescription { const char* ProjectId; int EngineType; const char* EngineVersion; };
struct ApplicationIdentifier {
    unsigned IdentifierType;
    union Value { ProjectDescription ProjectDesc; unsigned long long ApplicationId; } v;
};
struct FeatureDiscoveryInfo {
    unsigned SDKVersion, FeatureID;
    ApplicationIdentifier Identifier;
    const wchar_t* ApplicationDataPath;
    const FeatureCommonInfo* FeatureInfo;
};
struct FeatureRequirement { unsigned FeatureSupported, MinHWArchitecture; char MinOSVersion[255]; };
struct ProjectIdentity { std::string id; int engine; std::string version; };
static_assert(sizeof(FeatureCommonInfo) == 40 && sizeof(FeatureDiscoveryInfo) == 56 && sizeof(FeatureRequirement) == 264);
}
