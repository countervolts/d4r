# Sourced by the scripts that run the HIP SDK's clang. Nothing happens outside Windows (Git Bash / MSYS2).
# On Windows that clang reads the newest Visual Studio's C++ headers, which can be newer than it accepts:
# ROCm 7.14's clang rejects Visual Studio 18's <cmath> ("__device__ function 'isgreater' cannot overload
# __host__ __device__ function"). Unless VCToolsInstallDir is already set (a Developer prompt, or by hand),
# point it at Visual Studio 2022's MSVC toolset when one is installed.
case "$(uname -s)" in
MINGW*|MSYS*|CYGWIN*)
    if [[ -z "${VCToolsInstallDir:-}" ]]; then
        # Python on Windows passes copied environments with upper-case names (PROGRAMFILES(X86))
        d4r_x86="$(printenv 'ProgramFiles(x86)' || printenv 'PROGRAMFILES(X86)' || echo 'C:\Program Files (x86)')"
        d4r_vswhere="$d4r_x86/Microsoft Visual Studio/Installer/vswhere.exe"
        if [[ -x "$(cygpath -u "$d4r_vswhere" 2>/dev/null)" ]]; then
            d4r_vs="$("$(cygpath -u "$d4r_vswhere")" -version '[17.0,18.0)' -latest -property installationPath | tr -d '\r')"
            d4r_version_file="$(cygpath -u "$d4r_vs" 2>/dev/null)/VC/Auxiliary/Build/Microsoft.VCToolsVersion.default.txt"
            if [[ -n "$d4r_vs" && -f "$d4r_version_file" ]]; then
                export VCToolsInstallDir="$d4r_vs\\VC\\Tools\\MSVC\\$(tr -d '\r\n' < "$d4r_version_file")\\"
            fi
        fi
        unset d4r_x86 d4r_vswhere d4r_vs d4r_version_file
    fi
    ;;
esac
