#include <windows.h>
#include <dirent.h>
#include <dlfcn.h>
#include <errno.h>
#include <pthread.h>
#include <signal.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include "d4r_native_selection.h"
#include "d4r_motion_dilation.h"
#ifdef D4R_MOTION_KERNELS
#include "d4r_motion_kernels.h"
#endif

typedef int CUresult;
typedef int CUdevice;
typedef uint64_t CUdeviceptr;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUarray;
typedef void* CUmipmappedArray;
typedef void* CUstream;
typedef void* CUevent;
typedef uint64_t CUtexObject;
typedef uint64_t CUsurfObject;
typedef void* CUexternalMemory;

/* CUDA 12.8's NGX module passes the size_t-dimension array descriptor. */
typedef struct
{
    size_t Width;
    size_t Height;
    int32_t Format;
    uint32_t NumChannels;
} CUDA_ARRAY_DESCRIPTOR_V2;

typedef struct
{
    size_t Width;
    size_t Height;
    size_t Depth;
    int32_t Format;
    uint32_t NumChannels;
    uint32_t Flags;
} CUDA_ARRAY3D_DESCRIPTOR_V2;

enum
{
    CUDA_SUCCESS = 0,
    CUDA_ERROR_NOT_INITIALIZED = 3,
    CUDA_ERROR_INVALID_VALUE = 1,
    CUDA_ERROR_OUT_OF_MEMORY = 2,
    CUDA_ERROR_NOT_FOUND = 500,
    CUDA_ERROR_NOT_READY = 600,
    CUDA_ERROR_LAUNCH_FAILED = 719,
    CUDA_ERROR_NOT_SUPPORTED = 801,
    CUDA_ERROR_UNKNOWN = 999,
};

static void* cuda_library;
static int microcuda_k_requested;
static pthread_once_t load_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t trace_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t array_descriptor_lock = PTHREAD_MUTEX_INITIALIZER;
static pthread_mutex_t texture_descriptor_lock = PTHREAD_MUTEX_INITIALIZER;
static unsigned int capture_sequence;

typedef struct ArrayDescriptorRecord
{
    CUarray array;
    CUDA_ARRAY_DESCRIPTOR_V2 descriptor;
    struct ArrayDescriptorRecord* next;
} ArrayDescriptorRecord;

static ArrayDescriptorRecord* array_descriptors;

enum { CUDA_RESOURCE_DESC_BYTES = 144 };

typedef struct TextureObjectDescriptorRecord
{
    CUtexObject object;
    unsigned char descriptor[CUDA_RESOURCE_DESC_BYTES];
    struct TextureObjectDescriptorRecord* next;
} TextureObjectDescriptorRecord;

static TextureObjectDescriptorRecord* texture_object_descriptors;

static int remember_array_descriptor(CUarray array, const CUDA_ARRAY_DESCRIPTOR_V2* descriptor)
{
    ArrayDescriptorRecord* record = (ArrayDescriptorRecord*)malloc(sizeof(*record));
    if (record == NULL)
        return 0;
    record->array = array;
    record->descriptor = *descriptor;
    pthread_mutex_lock(&array_descriptor_lock);
    record->next = array_descriptors;
    array_descriptors = record;
    pthread_mutex_unlock(&array_descriptor_lock);
    return 1;
}

static void forget_array_descriptor(CUarray array)
{
    pthread_mutex_lock(&array_descriptor_lock);
    ArrayDescriptorRecord** current = &array_descriptors;
    while (*current != NULL)
    {
        if ((*current)->array == array)
        {
            ArrayDescriptorRecord* removed = *current;
            *current = removed->next;
            free(removed);
            break;
        }
        current = &(*current)->next;
    }
    pthread_mutex_unlock(&array_descriptor_lock);
}

static int lookup_array_descriptor(CUarray array, CUDA_ARRAY_DESCRIPTOR_V2* descriptor)
{
    int found = 0;
    pthread_mutex_lock(&array_descriptor_lock);
    for (ArrayDescriptorRecord* current = array_descriptors; current != NULL; current = current->next)
    {
        if (current->array == array)
        {
            *descriptor = current->descriptor;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&array_descriptor_lock);
    return found;
}

static int remember_texture_descriptor(CUtexObject object, const void* descriptor)
{
    TextureObjectDescriptorRecord* record =
        (TextureObjectDescriptorRecord*)malloc(sizeof(*record));
    if (record == NULL || descriptor == NULL)
    {
        free(record);
        return 0;
    }
    record->object = object;
    memcpy(record->descriptor, descriptor, CUDA_RESOURCE_DESC_BYTES);
    pthread_mutex_lock(&texture_descriptor_lock);
    record->next = texture_object_descriptors;
    texture_object_descriptors = record;
    pthread_mutex_unlock(&texture_descriptor_lock);
    return 1;
}

static void forget_texture_descriptor(CUtexObject object)
{
    pthread_mutex_lock(&texture_descriptor_lock);
    TextureObjectDescriptorRecord** current = &texture_object_descriptors;
    while (*current != NULL)
    {
        if ((*current)->object == object)
        {
            TextureObjectDescriptorRecord* removed = *current;
            *current = removed->next;
            free(removed);
            break;
        }
        current = &(*current)->next;
    }
    pthread_mutex_unlock(&texture_descriptor_lock);
}

static int lookup_texture_descriptor(CUtexObject object, void* descriptor)
{
    int found = 0;
    pthread_mutex_lock(&texture_descriptor_lock);
    for (TextureObjectDescriptorRecord* current = texture_object_descriptors;
         current != NULL; current = current->next)
    {
        if (current->object == object)
        {
            memcpy(descriptor, current->descriptor, CUDA_RESOURCE_DESC_BYTES);
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&texture_descriptor_lock);
    return found;
}

static void tracef(const char* format, ...);

/* --- portable installs --------------------------------------------------------------------------
   The drag-in release keeps ZLUDA and the native kernels in an d4r folder next to the game and
   runs under Steam's container runtime, whose library search path has no ROCm. The NGX shim reads
   that folder's d4r.ini and hands the Linux-side settings over with d4rSetEnv before its first
   CUDA call:
     D4R_ZLUDA_LIBCUDA     ZLUDA's libcuda.so
     D4R_ROCM_DIR          ROCm installation, searched before /opt/rocm. Its HIP libraries are
                           loaded by path; a dependency the search path lacks is looked up there
                           and in the host's library directories (/run/host inside the container).
     D4R_ZLUDA_CACHE_HOME  where ZLUDA keeps compiled kernels; XDG_CACHE_HOME is set to it only
                           while ZLUDA initialises
   A leading ~/ in these paths is the home directory.
   An D4R_ZLUDA_NATIVE_DIR holding d4r-kernels.txt ("NAME FNV1A64" lines: the hash of the PTX
   module each native kernel was written for) is not handed to ZLUDA directly. ZLUDA reads a
   per-process directory instead, and a native kernel is linked into it when NGX loads a module
   whose PTX matches, so a DLSS version with a changed kernel runs ZLUDA's own compile of it. */
static char load_error[768];

static void set_load_error(const char* format, ...)
{
    va_list args;
    va_start(args, format);
    vsnprintf(load_error, sizeof(load_error), format, args);
    va_end(args);
    tracef("%s", load_error);
}

/* The last library or initialisation failure, "" when there was none. */
const char* WINAPI d4rLoadError(void)
{
    return load_error;
}

/* Sets (value != NULL) or removes a Linux-side environment variable; overwrite = 0 keeps a value
   the process was started with. Returns 1 on success. */
int WINAPI d4rSetEnv(const char* name, const char* value, int overwrite)
{
    if (name == NULL || name[0] == '\0' || strchr(name, '=') != NULL)
        return 0;
    return (value != NULL ? setenv(name, value, overwrite) : unsetenv(name)) == 0;
}

static void expand_home(const char* path, char* out, size_t size)
{
    const char* home = getenv("HOME");
    if (path[0] == '~' && path[1] == '/' && home != NULL)
        snprintf(out, size, "%s%s", home, path + 1);
    else
        snprintf(out, size, "%s", path);
}

static void make_directories(const char* path)
{
    char partial[1024];
    snprintf(partial, sizeof(partial), "%s", path);
    for (char* slash = strchr(partial + 1, '/'); slash != NULL; slash = strchr(slash + 1, '/'))
    {
        *slash = '\0';
        mkdir(partial, 0755);
        *slash = '/';
    }
    mkdir(partial, 0755);
}

enum { LIBRARY_DIR_CAPACITY = 12 };
static char library_dirs[LIBRARY_DIR_CAPACITY][1024];
static int library_dir_count;

static void add_library_dir(const char* root, const char* suffix)
{
    if (library_dir_count < LIBRARY_DIR_CAPACITY)
        snprintf(library_dirs[library_dir_count++], sizeof(library_dirs[0]), "%s%s", root, suffix);
}

static void init_library_dirs(void)
{
    const char* rocm = getenv("D4R_ROCM_DIR");
    if (rocm != NULL && rocm[0] != '\0')
    {
        char expanded[1024];
        expand_home(rocm, expanded, sizeof(expanded));
        add_library_dir(expanded, "/lib");
        add_library_dir(expanded, "/lib/llvm/lib");
    }
    add_library_dir("/opt/rocm", "/lib");
    add_library_dir("/opt/rocm", "/lib/llvm/lib");
    /* the host's libraries as Steam's container runtime mounts them */
    static const char* const host[] = {"/run/host/usr/lib", "/run/host/usr/lib64",
                                       "/run/host/usr/lib/x86_64-linux-gnu", "/run/host/lib",
                                       "/run/host/lib64", "/run/host/lib/x86_64-linux-gnu"};
    for (size_t i = 0; i < sizeof(host) / sizeof(host[0]); ++i)
        add_library_dir(host[i], "");
}

static void* load_library_resolving(const char* path, int depth);

static int load_from_library_dirs(const char* name, int depth)
{
    for (int i = 0; i < library_dir_count; ++i)
    {
        char full[1280];
        snprintf(full, sizeof(full), "%s/%s", library_dirs[i], name);
        if (access(full, R_OK) == 0 && load_library_resolving(full, depth) != NULL)
            return 1;
    }
    return 0;
}

/* dlopen that loads a missing dependency from the library directories and retries: the later
   load finds it by its soname. */
static void* load_library_resolving(const char* path, int depth)
{
    for (int attempt = 0; attempt < 32; ++attempt)
    {
        void* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
        if (handle != NULL)
            return handle;
        const char* error = dlerror();
        const char* end = error != NULL ? strstr(error, ": cannot open shared object file") : NULL;
        char missing_name[256];
        const size_t length = end != NULL ? (size_t)(end - error) : 0;
        if (length == 0 || length >= sizeof(missing_name) || depth >= 8)
        {
            set_load_error("dlopen(%s) failed: %s", path, error != NULL ? error : "unknown error");
            return NULL;
        }
        memcpy(missing_name, error, length);
        missing_name[length] = '\0';
        if (strchr(missing_name, '/') != NULL || !load_from_library_dirs(missing_name, depth + 1))
        {
            set_load_error("dlopen(%s) failed: %s is not in the search path, ROCm or the host's library "
                           "directories", path, missing_name);
            return NULL;
        }
    }
    set_load_error("dlopen(%s) failed: too many missing dependencies", path);
    return NULL;
}

/* HIP's runtime and the comgr library HIP opens by name later (ZLUDA itself links only
   libamdhip64). An D4R_ROCM_DIR copy is preferred over one the search path would find. */
static void preload_rocm(void)
{
    static const char* const libraries[][2] = {
        {"libhsa-runtime64.so.1", NULL}, {"libamd_comgr.so.3", "libamd_comgr.so.2"}, {"libamdhip64.so.7", NULL}};
    for (size_t i = 0; i < sizeof(libraries) / sizeof(libraries[0]); ++i)
    {
        void* handle = NULL;
        for (int choice = 0; choice < 2 && handle == NULL && libraries[i][choice] != NULL; ++choice)
        {
            const char* name = libraries[i][choice];
            for (int d = 0; d < library_dir_count && d < 2 && handle == NULL; ++d)
            {
                char full[1280];
                snprintf(full, sizeof(full), "%s/%s", library_dirs[d], name);
                if (getenv("D4R_ROCM_DIR") != NULL && access(full, R_OK) == 0)
                    handle = load_library_resolving(full, 0);
            }
            if (handle == NULL)
                handle = load_library_resolving(name, 0);
        }
        tracef("ROCm %s: %s", libraries[i][0], handle != NULL ? "loaded" : load_error);
    }
}

typedef struct
{
    char name[128];
    uint64_t hash;
    int linked;
} NativeKernel;

static NativeKernel* native_kernels;
static size_t native_kernel_count;
static char native_source[1024];
static char native_served[1024];
static pthread_mutex_t native_lock = PTHREAD_MUTEX_INITIALIZER;

static void remove_directory(const char* path)
{
    DIR* directory = opendir(path);
    if (directory == NULL)
        return;
    for (struct dirent* entry; (entry = readdir(directory)) != NULL;)
    {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;
        char child[1400];
        snprintf(child, sizeof(child), "%s/%s", path, entry->d_name);
        unlink(child);
    }
    closedir(directory);
    rmdir(path);
}

static void native_cleanup(void)
{
    if (native_served[0] != '\0')
        remove_directory(native_served);
}

/* The gfx target ("gfx1101") of the GPU DLSS runs on, "" when unknown: D4R_GPU_ARCH if set, else the KFD
   topology's GPU with the most SIMDs. With an integrated and a discrete GPU (Ryzen 7000/9000 desktops) the
   first KFD GPU node can be the integrated one, whose kernels would not match the discrete GPU. */
static void gpu_architecture(char* out, size_t size)
{
    out[0] = '\0';
    const char* forced = getenv("D4R_GPU_ARCH");
    if (forced != NULL && strncmp(forced, "gfx", 3) == 0)
    {
        snprintf(out, size, "%s", forced);
        return;
    }
    unsigned long best_simds = 0;
    int gpus = 0;
    for (int node = 0; node < 16; ++node)
    {
        char path[128];
        snprintf(path, sizeof(path), "/sys/class/kfd/kfd/topology/nodes/%d/properties", node);
        FILE* properties = fopen(path, "r");
        if (properties == NULL)
            continue;
        char line[256];
        unsigned long simds = 0, target = 0;
        while (fgets(line, sizeof(line), properties) != NULL)
        {
            sscanf(line, "simd_count %lu", &simds);
            sscanf(line, "gfx_target_version %lu", &target);
        }
        fclose(properties);
        if (simds == 0 || target == 0)
            continue;
        ++gpus;
        if (simds > best_simds)
        {
            best_simds = simds;
            snprintf(out, size, "gfx%lu%lu%lx", target / 10000, (target / 100) % 100, target % 100);
        }
    }
    if (gpus > 1)
        tracef("native kernels: %d GPUs; using the one with the most SIMDs (%s); D4R_GPU_ARCH overrides", gpus, out);
}

static void prepare_native_kernels(const char* cache_home)
{
    const char* configured = getenv("D4R_ZLUDA_NATIVE_DIR");
    if (configured == NULL || configured[0] == '\0')
        return;
    /* a release directory holds one manifest and set of code objects per GPU target */
    char architecture[32], source[1024], path[1200];
    gpu_architecture(architecture, sizeof(architecture));
    FILE* manifest = NULL;
    /* RDNA4 with native FP8 WMMA (D4R_ZLUDA_WMMA_FP8_NATIVE=1, d4r.ini NativeFp8): the <target>-fp8 variant,
       whose kernels match ZLUDA's FP8 lowering; no other GPU has such a directory */
    const char* fp8 = getenv("D4R_ZLUDA_WMMA_FP8_NATIVE");
    const char* prefer = getenv("D4R_PREFER_ACCURACY");
    const int accuracy = prefer != NULL && strcmp(prefer, "1") == 0;
    const int kind = d4r_select_native_source(configured, architecture, accuracy,
        fp8 != NULL && strcmp(fp8, "1") == 0, source, sizeof(source));
    if (kind == 0)
    {
        tracef("native kernels: no %sset for this GPU (%s) in %s; ZLUDA compiles every DLSS kernel",
               accuracy ? "accuracy " : "", architecture[0] != '\0' ? architecture : "unknown target", configured);
        unsetenv("D4R_ZLUDA_NATIVE_DIR");
        return;
    }
    if (kind == 2)
    {
        /* A flat developer set has no per-kernel manifest to filter. Fail safely to
           translated kernels rather than silently retaining the disabled encoders. */
        if (!d4r_native_swin_encoders_enabled())
        {
            tracef("NativeSwinEncoders off: developer set needs d4r-kernels.txt for selective fallback; "
                   "using translated kernels for this set");
            unsetenv("D4R_ZLUDA_NATIVE_DIR");
            return;
        }
        setenv("D4R_ZLUDA_NATIVE_DIR", source, 1);
        tracef("native kernels: %sdeveloper set in %s", accuracy ? "accuracy " : "", source);
        return;
    }
    snprintf(path, sizeof(path), "%s/d4r-kernels.txt", source);
    manifest = cache_home != NULL ? fopen(path, "r") : NULL; /* a manifest set is served from the cache */
    if (manifest == NULL)
    {
        unsetenv("D4R_ZLUDA_NATIVE_DIR");
        return;
    }
    char line[512];
    size_t capacity = 0;
    while (fgets(line, sizeof(line), manifest) != NULL)
    {
        char name[128];
        unsigned long long hash = 0;
        if (line[0] == '#' || sscanf(line, "%127s %llx", name, &hash) != 2)
            continue;
        if (!d4r_native_kernel_allowed(name))
            continue;
        if (native_kernel_count == capacity)
        {
            capacity = capacity != 0 ? capacity * 2 : 32;
            NativeKernel* grown = (NativeKernel*)realloc(native_kernels, capacity * sizeof(NativeKernel));
            if (grown == NULL)
                break;
            native_kernels = grown;
        }
        NativeKernel* kernel = &native_kernels[native_kernel_count++];
        snprintf(kernel->name, sizeof(kernel->name), "%s", name);
        kernel->hash = hash;
        kernel->linked = 0;
    }
    fclose(manifest);
    if (!d4r_native_swin_encoders_enabled())
        tracef("NativeSwinEncoders off: rrlite_enc1_4x4 and rrlite_enc2_4x4 use original translated PTX; "
               "other native kernels retained");

    /* <cache>/d4r-native/<pid>; directories of processes that no longer exist are removed */
    char base[1024];
    snprintf(base, sizeof(base), "%s/d4r-native", cache_home);
    make_directories(base);
    DIR* directory = opendir(base);
    if (directory != NULL)
    {
        for (struct dirent* entry; (entry = readdir(directory)) != NULL;)
        {
            char* end = NULL;
            const long pid = strtol(entry->d_name, &end, 10);
            if (pid > 0 && end != NULL && *end == '\0' && kill((pid_t)pid, 0) != 0 && errno == ESRCH)
            {
                char stale[1400];
                snprintf(stale, sizeof(stale), "%s/%s", base, entry->d_name);
                remove_directory(stale);
            }
        }
        closedir(directory);
    }
    snprintf(native_served, sizeof(native_served), "%s/%ld", base, (long)getpid());
    remove_directory(native_served);
    char* resolved = realpath(source, NULL);
    if (resolved == NULL || strlen(resolved) >= sizeof(native_source) || mkdir(native_served, 0755) != 0)
    {
        free(resolved);
        set_load_error("cannot create %s from %s; native kernels disabled", native_served, source);
        native_served[0] = '\0';
        native_kernel_count = 0;
        unsetenv("D4R_ZLUDA_NATIVE_DIR");
        return;
    }
    snprintf(native_source, sizeof(native_source), "%s", resolved);
    free(resolved);
    setenv("D4R_ZLUDA_NATIVE_DIR", native_served, 1);
    atexit(native_cleanup);
    tracef("native kernels: %zu in %s, each served from %s once DLSS's PTX for it matches",
           native_kernel_count, native_source, native_served);
}

/* LZ4 block decoder for compressed fatbin entries; returns the decoded size, 0 on bad input. */
static size_t lz4_block(const unsigned char* src, size_t src_size, unsigned char* dst, size_t dst_size)
{
    size_t i = 0, o = 0;
    while (i < src_size)
    {
        const unsigned int token = src[i++];
        size_t literals = token >> 4;
        if (literals == 15)
        {
            unsigned char b;
            do
            {
                if (i >= src_size)
                    return 0;
                b = src[i++];
                literals += b;
            } while (b == 255);
        }
        if (literals > src_size - i || literals > dst_size - o)
            return 0;
        memcpy(dst + o, src + i, literals);
        i += literals;
        o += literals;
        if (i >= src_size || o >= dst_size)
            break;
        if (src_size - i < 2)
            return 0;
        const size_t offset = src[i] | ((size_t)src[i + 1] << 8);
        i += 2;
        if (offset == 0 || offset > o)
            return 0;
        size_t match = token & 15;
        if (match == 15)
        {
            unsigned char b;
            do
            {
                if (i >= src_size)
                    return 0;
                b = src[i++];
                match += b;
            } while (b == 255);
        }
        match += 4;
        if (match > dst_size - o)
            match = dst_size - o;
        for (size_t k = 0; k < match; ++k, ++o)
            dst[o] = dst[o - offset];
    }
    return o;
}

static int is_ptx_name_char(unsigned char c)
{
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '_' || c == '$';
}

/* Links the native kernel of every .entry of one PTX module whose text matches the manifest. */
static void verify_ptx_module(const unsigned char* text, size_t size)
{
    while (size > 0 && text[size - 1] == '\0')
        --size;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < size; ++i)
        hash = (hash ^ text[i]) * 0x100000001b3ull;
    for (size_t i = 0; i + 7 < size; ++i)
    {
        if (memcmp(text + i, ".entry", 6) != 0 || (text[i + 6] != ' ' && text[i + 6] != '\t' && text[i + 6] != '\n'))
            continue;
        size_t p = i + 6;
        while (p < size && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r'))
            ++p;
        const size_t start = p;
        while (p < size && is_ptx_name_char(text[p]))
            ++p;
        const size_t length = p - start;
        while (p < size && (text[p] == ' ' || text[p] == '\t' || text[p] == '\n' || text[p] == '\r'))
            ++p;
        if (length == 0 || length >= sizeof(native_kernels[0].name) || p >= size || text[p] != '(')
            continue;
        char name[128];
        memcpy(name, text + start, length);
        name[length] = '\0';
        int known = 0, matched = 0;
        uint64_t expected = 0;
        for (size_t k = 0; k < native_kernel_count; ++k)
        {
            if (strcmp(native_kernels[k].name, name) != 0)
                continue;
            known = 1;
            expected = native_kernels[k].hash;
            if (native_kernels[k].hash == hash)
            {
                matched = 1;
                if (!native_kernels[k].linked)
                {
                    char target[1400], link_path[1400];
                    snprintf(target, sizeof(target), "%s/%s.hsaco", native_source, name);
                    snprintf(link_path, sizeof(link_path), "%s/%s.hsaco", native_served, name);
                    if (symlink(target, link_path) == 0 || errno == EEXIST)
                        native_kernels[k].linked = 1;
                    tracef("native kernel %s: %s", name, native_kernels[k].linked ? "verified" : "link failed");
                }
            }
        }
        if (known && !matched)
            tracef("native kernel %s not used: this DLSS's PTX for it differs (%016llx, expected %016llx)", name,
                   (unsigned long long)hash, (unsigned long long)expected);
        i = p;
    }
}

static void verify_native_kernels(const void* image)
{
    if (native_kernel_count == 0 || image == NULL)
        return;
    const unsigned char* bytes = (const unsigned char*)image;
    uint32_t magic = 0;
    memcpy(&magic, bytes, sizeof(magic));
    pthread_mutex_lock(&native_lock);
    if (magic != 0xba55ed50u)
    {
        /* a PTX text image */
        const unsigned char* end = (const unsigned char*)memchr(bytes, 0, 64u * 1024u * 1024u);
        if (end != NULL)
            verify_ptx_module(bytes, (size_t)(end - bytes));
        pthread_mutex_unlock(&native_lock);
        return;
    }
    uint16_t version = 0, header_size = 0;
    uint64_t files_size = 0;
    memcpy(&version, bytes + 4, sizeof(version));
    memcpy(&header_size, bytes + 6, sizeof(header_size));
    memcpy(&files_size, bytes + 8, sizeof(files_size));
    if (version == 1 && header_size >= 16 && header_size <= 4096 && files_size <= 256u * 1024u * 1024u)
    {
        size_t offset = header_size;
        const size_t end = (size_t)header_size + (size_t)files_size;
        while (offset + 64 <= end)
        {
            uint16_t kind = 0;
            uint32_t entry_header = 0;
            uint64_t entry_size = 0, flags = 0, decompressed = 0;
            memcpy(&kind, bytes + offset, sizeof(kind));
            memcpy(&entry_header, bytes + offset + 4, sizeof(entry_header));
            memcpy(&entry_size, bytes + offset + 8, sizeof(entry_size));
            memcpy(&flags, bytes + offset + 40, sizeof(flags));
            if (entry_header >= 64)
                memcpy(&decompressed, bytes + offset + 56, sizeof(decompressed));
            if (entry_header < 16 || entry_header > end - offset || entry_size > end - offset - entry_header)
                break;
            const unsigned char* payload = bytes + offset + entry_header;
            if (kind == 1) /* PTX */
            {
                if ((flags & 0x2000) != 0 && decompressed != 0 && decompressed <= 256u * 1024u * 1024u)
                {
                    unsigned char* text = (unsigned char*)malloc((size_t)decompressed);
                    const size_t size = text != NULL ? lz4_block(payload, (size_t)entry_size, text, (size_t)decompressed) : 0;
                    if (size != 0)
                        verify_ptx_module(text, size);
                    free(text);
                }
                else
                    verify_ptx_module(payload, (size_t)entry_size);
            }
            offset += entry_header + entry_size;
        }
    }
    pthread_mutex_unlock(&native_lock);
}

/* DLSS's output kernels (hiluma_engine_output_*, rrlite_downsample_kernel_*): NGX looks up every variant, then launches the one
   matching the feature's flags. The shim's direct output relies on stores that only d4r's native
   output kernel makes, so it asks whether the last output kernel launched was native (a file in
   D4R_ZLUDA_NATIVE_DIR, which is the verified per-process directory in a portable install). */
enum { OUTPUT_KERNEL_CAPACITY = 64 };
static struct
{
    CUfunction function;
    int native;
} output_kernels[OUTPUT_KERNEL_CAPACITY]; /* the latest lookups, newest last */
static pthread_mutex_t output_kernel_lock = PTHREAD_MUTEX_INITIALIZER;
static int last_output_native = -1;

static void note_function_lookup(CUfunction function, const char* name)
{
    /* DLSS 4 writes its result with hiluma_engine_output_*, DLSS 4.5 (rrlite) with its final
       rrlite_downsample_kernel_* (the post kernel works at a larger internal resolution) */
    if (name == NULL || (strncmp(name, "hiluma_engine_output", 20) != 0 && strncmp(name, "rrlite_downsample_kernel", 24) != 0))
        return;
    const char* directory = getenv("D4R_ZLUDA_NATIVE_DIR");
    int native = 0;
    if (directory != NULL && directory[0] != '\0')
    {
        char path[1400];
        snprintf(path, sizeof(path), "%s/%s.hsaco", directory, name);
        native = access(path, R_OK) == 0;
    }
    pthread_mutex_lock(&output_kernel_lock);
    memmove(output_kernels, output_kernels + 1, sizeof(output_kernels) - sizeof(output_kernels[0]));
    output_kernels[OUTPUT_KERNEL_CAPACITY - 1].function = function;
    output_kernels[OUTPUT_KERNEL_CAPACITY - 1].native = native;
    pthread_mutex_unlock(&output_kernel_lock);
}

static void note_launch(CUfunction function)
{
    pthread_mutex_lock(&output_kernel_lock);
    /* newest first, so a recycled handle resolves with its latest lookup */
    for (int i = OUTPUT_KERNEL_CAPACITY - 1; i >= 0 && output_kernels[i].function != NULL; --i)
        if (output_kernels[i].function == function)
        {
            __atomic_store_n(&last_output_native, output_kernels[i].native, __ATOMIC_RELAXED);
            break;
        }
    pthread_mutex_unlock(&output_kernel_lock);
}

/* 1 when the last output kernel launched was d4r's native one, 0 when it was ZLUDA's compile of
   NVIDIA's, -1 before the first. */
int WINAPI d4rOutputKernelNative(void)
{
    return __atomic_load_n(&last_output_native, __ATOMIC_RELAXED);
}

static void load_zluda(void)
{
    /* Explicit experimental backend. Never silently mix module/context/array
       handles from MicroCUDA and ZLUDA, and never fall back after selection. */
    const char* backend = getenv("D4R_CUDA_BACKEND");
    if (backend != NULL && strcmp(backend, "microcuda-k") == 0)
    {
        microcuda_k_requested = 1;
        backend = "zluda";
        tracef("microcuda-k selected: audited K enc0 via MicroCUDA; all remaining APIs/kernels via ZLUDA (partial integration)");
    }
    if (backend != NULL && backend[0] != '\0' && strcmp(backend, "zluda") != 0)
    {
        if (strcmp(backend, "microcuda") != 0)
        {
            set_load_error("unknown D4R_CUDA_BACKEND: %s", backend);
            return;
        }
        const char* library = getenv("D4R_MICROCUDA_LIBCUDA");
        if (library == NULL || library[0] == '\0')
        {
            set_load_error("microcuda requires D4R_MICROCUDA_LIBCUDA and D4R_MICROCUDA_PACK");
            return;
        }
        const char* prefer = getenv("D4R_PREFER_ACCURACY");
        if (prefer != NULL && strcmp(prefer, "1") == 0)
        {
            set_load_error("microcuda has no PreferAccuracy module pack");
            return;
        }
        char native_path[1024];
        expand_home(library, native_path, sizeof(native_path));
        init_library_dirs();
        preload_rocm();
        cuda_library = load_library_resolving(native_path, 0);
        if (cuda_library != NULL)
        {
            load_error[0] = '\0';
            /* Native kernels are verified and served exactly as for ZLUDA (per-process directory,
               linked when NGX loads a module whose PTX matches); MicroCUDA reads that directory. */
            const char* cache = getenv("D4R_ZLUDA_CACHE_HOME");
            char cache_home[1024] = {0};
            if (cache != NULL && cache[0] != '\0')
            {
                expand_home(cache, cache_home, sizeof(cache_home));
                make_directories(cache_home);
            }
            prepare_native_kernels(cache_home[0] != '\0' ? cache_home : NULL);
            const char* served = getenv("D4R_ZLUDA_NATIVE_DIR");
            if (served != NULL && served[0] != '\0')
                setenv("D4R_MICROCUDA_NATIVE_DIR", served, 1);
            else
                unsetenv("D4R_MICROCUDA_NATIVE_DIR");
            tracef("MicroCUDA backend selected: %s (modules from D4R_MICROCUDA_PACK; no ZLUDA fallback)", native_path);
        }
        return;
    }
    /* A policy switch, applied before ZLUDA loads or reads its compilation/cache settings. This
       also covers native-off and missing/version-mismatched native kernels. */
    const int accuracy = d4r_apply_accuracy_policy();
    if (accuracy)
    {
        tracef("PreferAccuracy on: original denormal handling, per-MMA rounding, wave32 and NGX synchronizations");
    }
    /* D4R_ZLUDA_LIBCUDA may name ZLUDA's libcuda.so directly, for processes
       (such as Proton games) whose library search path is not ours. */
    const char* configured = getenv("D4R_ZLUDA_LIBCUDA");
    char path[1024];
    expand_home(configured != NULL && configured[0] != '\0' ? configured : "libcuda.so", path, sizeof(path));
    init_library_dirs();
    if (getenv("D4R_ROCM_DIR") != NULL)
        preload_rocm();
    cuda_library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (cuda_library == NULL)
    {
        const char* error = dlerror();
        tracef("dlopen(%s): %s; loading ROCm from its installation", path, error != NULL ? error : "failed");
        preload_rocm();
        cuda_library = load_library_resolving(path, 0);
    }
    if (cuda_library == NULL)
    {
        fprintf(stderr, "[d4r nvcuda] %s\n", load_error);
        return;
    }
    load_error[0] = '\0';
    const char* cache = getenv("D4R_ZLUDA_CACHE_HOME");
    const char* xdg = getenv("XDG_CACHE_HOME");
    const char* home = getenv("HOME");
    char cache_home[1024] = {0}, root[1024] = {0};
    if (cache != NULL && cache[0] != '\0')
    {
        expand_home(cache, cache_home, sizeof(cache_home));
        make_directories(cache_home);
    }
    /* the cache root; no /tmp fallback, since other users can plant links there */
    if (cache_home[0] != '\0')
        snprintf(root, sizeof(root), "%s", cache_home);
    else if (xdg != NULL && xdg[0] != '\0')
        snprintf(root, sizeof(root), "%s", xdg);
    else if (home != NULL)
        snprintf(root, sizeof(root), "%s/.cache", home);
    if (accuracy)
    {
        /* FAST_MATH is not fingerprinted by older ZLUDA runtimes. Use a separate cache even
           when a previous run explicitly enabled that experimental compiler switch. */
        if (snprintf(cache_home, sizeof(cache_home), "%s/d4r-accuracy", root[0] != '\0' ? root : "/tmp/.cache") >= (int)sizeof(cache_home))
        {
            set_load_error("accuracy cache path is too long");
            return;
        }
        make_directories(cache_home);
    }
    prepare_native_kernels(root[0] == '\0' ? NULL : accuracy ? cache_home : root);
    if (cache_home[0] != '\0')
    {
        /* ZLUDA picks its cache directory once, in cuInit */
        const char* previous = getenv("XDG_CACHE_HOME");
        char* saved = previous != NULL ? strdup(previous) : NULL;
        setenv("XDG_CACHE_HOME", cache_home, 1);
        typedef CUresult(__attribute__((sysv_abi)) * INIT_FN)(unsigned int);
        INIT_FN init = (INIT_FN)dlsym(cuda_library, "cuInit");
        const CUresult result = init != NULL ? init(0) : CUDA_ERROR_NOT_INITIALIZED;
        if (saved != NULL)
            setenv("XDG_CACHE_HOME", saved, 1);
        else
            unsetenv("XDG_CACHE_HOME");
        free(saved);
        if (result != CUDA_SUCCESS)
            set_load_error("ZLUDA cuInit failed (%d): is the ROCm HIP runtime installed and the GPU supported?", result);
        else
            tracef("ZLUDA initialised; kernel cache in %s", cache_home);
    }
}

static void* find_zluda_symbol(const char* name)
{
    pthread_once(&load_once, load_zluda);
    return cuda_library != NULL ? dlsym(cuda_library, name) : NULL;
}

/* d4r-owned, opt-in preprocessing. Runs on HIP's legacy/default stream, shared
   with CUDA's null stream; the shim records and waits for completion before NGX. */
CUresult WINAPI d4rDilateMotionVectors(const D4rMotionDilationParams* parameters)
{
    if (parameters == NULL || parameters->depth == 0 || parameters->motion == 0 || parameters->output == 0 ||
        parameters->radius < 1 || parameters->radius > 2 || parameters->render_width == 0 ||
        parameters->render_height == 0 || parameters->motion_width == 0 || parameters->motion_height == 0 ||
        parameters->depth_pitch < (uint64_t)parameters->render_width * 4 ||
        parameters->motion_pitch < (uint64_t)parameters->motion_width * 4 ||
        (uint64_t)parameters->motion_width * parameters->motion_height > 0x7fffffffull)
        return CUDA_ERROR_INVALID_VALUE;
#ifdef D4R_MOTION_KERNELS
    typedef int(__attribute__((sysv_abi)) * HIP_LOAD)(void**, const void*);
    typedef int(__attribute__((sysv_abi)) * HIP_FUNCTION)(void**, void*, const char*);
    typedef int(__attribute__((sysv_abi)) * HIP_LAUNCH)(void*, unsigned, unsigned, unsigned,
        unsigned, unsigned, unsigned, unsigned, void*, void**, void**);
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    static void* module;
    static void* function;
    static HIP_LAUNCH launch;
    static int initialized, result = CUDA_ERROR_NOT_SUPPORTED;
    pthread_mutex_lock(&lock);
    if (!initialized)
    {
        initialized = 1;
        pthread_once(&load_once, load_zluda);
        void* hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_LOCAL);
        char architecture[32];
        gpu_architecture(architecture, sizeof(architecture));
        const void* image = d4r_motion_image(architecture);
        HIP_LOAD load = hip != NULL ? (HIP_LOAD)dlsym(hip, "hipModuleLoadData") : NULL;
        HIP_FUNCTION get = hip != NULL ? (HIP_FUNCTION)dlsym(hip, "hipModuleGetFunction") : NULL;
        launch = hip != NULL ? (HIP_LAUNCH)dlsym(hip, "hipModuleLaunchKernel") : NULL;
        if (image != NULL && load != NULL && get != NULL && launch != NULL &&
            load(&module, image) == 0 && get(&function, module, "d4r_motion_dilate") == 0)
            result = CUDA_SUCCESS;
        tracef("motion-vector dilation: %s (%s)", result == 0 ? "GPU kernel ready" : "unavailable", architecture);
    }
    const int status = result;
    pthread_mutex_unlock(&lock);
    if (status != CUDA_SUCCESS)
        return status;
    D4rMotionDilationParams copy = *parameters;
    void* arguments[] = {&copy};
    const uint64_t pixels = (uint64_t)copy.motion_width * copy.motion_height;
    return launch(function, (unsigned)((pixels + 127) / 128), 1, 1, 128, 1, 1, 0, NULL, arguments, NULL);
#else
    return CUDA_ERROR_NOT_SUPPORTED;
#endif
}

static void tracef(const char* format, ...)
{
    char line[2048];
    va_list args;
    va_start(args, format);
    vsnprintf(line, sizeof(line), format, args);
    va_end(args);

    pthread_mutex_lock(&trace_lock);
    fprintf(stderr, "[d4r nvcuda] %s\n", line);
    fflush(stderr);

    char path[MAX_PATH] = {0};
    DWORD length = GetEnvironmentVariableA("D4R_CUDA_TRACE", path, sizeof(path));
    if (length > 0 && length < sizeof(path))
    {
        FILE* output = fopen(path, "a");
        if (output != NULL)
        {
            fprintf(output, "%s\n", line);
            fclose(output);
        }
    }
    pthread_mutex_unlock(&trace_lock);
}

/* Per-call tracing of hot-path APIs (launches, copies, events, syncs) is
   opt-in with D4R_CUDA_VERBOSE=1: writing ~400 flushed lines per DLSS
   evaluation on the submitting thread costs milliseconds per frame.
   Failures are always traced. */
static int trace_verbose(void)
{
    static int verbose = -1;
    if (verbose < 0)
    {
        const char* value = getenv("D4R_CUDA_VERBOSE");
        verbose = value != NULL && value[0] == '1';
    }
    return verbose;
}

#define TRACE_CALL(result, ...) \
    do \
    { \
        if ((result) != 0 || trace_verbose()) \
            tracef(__VA_ARGS__); \
    } while (0)

/* Wine exposes the PE side with the Windows ABI; ZLUDA's libcuda.so uses SysV. */
typedef CUresult(__attribute__((sysv_abi)) *CUINIT_FN)(unsigned int);
typedef CUresult(__attribute__((sysv_abi)) *CUDEVICEGETCOUNT_FN)(int*);
typedef CUresult(__attribute__((sysv_abi)) *CUDEVICEGET_FN)(CUdevice*, int);
typedef CUresult(__attribute__((sysv_abi)) *CUCTXCREATE_FN)(CUcontext*, unsigned int, CUdevice);
typedef CUresult(__attribute__((sysv_abi)) *CUDEVICEGETLUID_FN)(char*, unsigned int*, CUdevice);
typedef CUresult(__attribute__((sysv_abi)) *CUDEVICEGETATTRIBUTE_FN)(int*, int, CUdevice);
typedef CUresult(__attribute__((sysv_abi)) *CUDEVICEGETUUID_FN)(void*, CUdevice);
typedef CUresult(__attribute__((sysv_abi)) *CUCTX_CURRENT_FN)(CUcontext);
typedef CUresult(__attribute__((sysv_abi)) *CUCTX_POP_CURRENT_FN)(CUcontext*);
typedef CUresult(__attribute__((sysv_abi)) *CUCTX_GET_DEVICE_FN)(CUdevice*);
typedef CUresult(__attribute__((sysv_abi)) *CUCTXDESTROY_FN)(CUcontext);
typedef CUresult(__attribute__((sysv_abi)) *CUCTX_SYNCHRONIZE_FN)(void);
typedef CUresult(__attribute__((sysv_abi)) *CUMODULELOADDATA_FN)(CUmodule*, const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUMODULELOADDATAEX_FN)(CUmodule*, const void*, unsigned int, int*, void**);
typedef CUresult(__attribute__((sysv_abi)) *CUMODULELOAD_FN)(CUmodule*, const char*);
typedef CUresult(__attribute__((sysv_abi)) *CUMODULEUNLOAD_FN)(CUmodule);
typedef CUresult(__attribute__((sysv_abi)) *CUMODULEGETFUNCTION_FN)(CUfunction*, CUmodule, const char*);
typedef CUresult(__attribute__((sysv_abi)) *CULAUNCHKERNEL_FN)(
    CUfunction, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int, unsigned int,
    unsigned int, CUstream, void**, void**);
typedef CUresult(__attribute__((sysv_abi)) *CUEVENTCREATE_FN)(CUevent*, unsigned int);
typedef CUresult(__attribute__((sysv_abi)) *CUEVENTRECORD_FN)(CUevent, CUstream);
typedef CUresult(__attribute__((sysv_abi)) *CUEVENTELAPSED_FN)(float*, CUevent, CUevent);
typedef CUresult(__attribute__((sysv_abi)) *CUEVENTDESTROY_FN)(CUevent);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMALLOC_FN)(CUdeviceptr*, size_t);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMFREE_FN)(CUdeviceptr);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMALLOCHOST_FN)(void**, size_t);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMFREEHOST_FN)(void*);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMCPY2D_FN)(const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMCPYHTODASYNC_FN)(CUdeviceptr, const void*, size_t, CUstream);
typedef CUresult(__attribute__((sysv_abi)) *CUMEMCPYDTOH_FN)(void*, CUdeviceptr, size_t);
typedef CUresult(__attribute__((sysv_abi)) *CUARRAYCREATEV2_FN)(CUarray*, const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUARRAY3DCREATEV2_FN)(CUarray*, const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUARRAYDESTROY_FN)(CUarray);
typedef CUresult(__attribute__((sysv_abi)) *CUARRAYGETDESCRIPTORV2_FN)(void*, CUarray);
typedef CUresult(__attribute__((sysv_abi)) *CUMIPMAPPEDARRAYDESTROY_FN)(CUmipmappedArray);
typedef CUresult(__attribute__((sysv_abi)) *CUEXTERNALMEMORYDESTROY_FN)(CUexternalMemory);
typedef CUresult(__attribute__((sysv_abi)) *CUSURFOBJECTCREATE_FN)(CUsurfObject*, const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUSURFOBJECTDESTROY_FN)(CUsurfObject);
typedef CUresult(__attribute__((sysv_abi)) *CUSURFOBJECTGETDESC_FN)(void*, CUsurfObject);
typedef CUresult(__attribute__((sysv_abi)) *CUTEXOBJECTCREATE_FN)(CUtexObject*, const void*, const void*, const void*);
typedef CUresult(__attribute__((sysv_abi)) *CUTEXOBJECTDESTROY_FN)(CUtexObject);
typedef CUresult(__attribute__((sysv_abi)) *CUTEXOBJECTGETDESC_FN)(void*, CUtexObject);
typedef CUresult(__attribute__((sysv_abi)) *CUGETERRORSTRING_FN)(CUresult, const char**);
typedef CUresult(__attribute__((sysv_abi)) *CUGETPROCADDRESS_FN)(const char*, void**, int, uint64_t, int*);

static pthread_once_t context_once = PTHREAD_ONCE_INIT;
static CUresult context_setup_result = CUDA_ERROR_NOT_INITIALIZED;
static CUcontext cuda_context;

static unsigned int get_process_u32(const char* name, unsigned int fallback)
{
    char value[32] = {0};
    DWORD length = GetEnvironmentVariableA(name, value, sizeof(value));
    if (length == 0 || length >= sizeof(value))
        return fallback;
    char* end = NULL;
    unsigned long parsed = strtoul(value, &end, 0);
    return end != value && *end == '\0' ? (unsigned int)parsed : fallback;
}

static void create_default_zluda_context(void)
{
    CUINIT_FN init = (CUINIT_FN)find_zluda_symbol("cuInit");
    CUDEVICEGETCOUNT_FN get_device_count = (CUDEVICEGETCOUNT_FN)find_zluda_symbol("cuDeviceGetCount");
    CUDEVICEGET_FN get_device = (CUDEVICEGET_FN)find_zluda_symbol("cuDeviceGet");
    CUCTXCREATE_FN create_context = (CUCTXCREATE_FN)find_zluda_symbol("cuCtxCreate_v2");
    if (init == NULL || get_device_count == NULL || get_device == NULL || create_context == NULL)
    {
        tracef("ZLUDA is missing a context bootstrap export");
        return;
    }

    context_setup_result = init(0);
    if (context_setup_result != CUDA_SUCCESS)
    {
        tracef("cuInit returned %d", context_setup_result);
        return;
    }

    int device_count = 0;
    context_setup_result = get_device_count(&device_count);
    if (context_setup_result != CUDA_SUCCESS || device_count < 1)
    {
        tracef("cuDeviceGetCount returned %d, count %d", context_setup_result, device_count);
        if (context_setup_result == CUDA_SUCCESS)
            context_setup_result = CUDA_ERROR_NOT_INITIALIZED;
        return;
    }

    CUdevice device = 0;
    context_setup_result = get_device(&device, 0);
    if (context_setup_result == CUDA_SUCCESS)
        context_setup_result = create_context(&cuda_context, 0, device);
    if (context_setup_result != CUDA_SUCCESS)
    {
        tracef("context creation returned %d", context_setup_result);
        return;
    }

    tracef("created CUDA context for device %d", device);
}

static void ensure_context(void)
{
    pthread_once(&context_once, create_default_zluda_context);
}

/* Opt-in stage instrumentation. With D4R_CUDA_LAUNCH_STATS=1 every
   cuLaunchKernel is followed by a context synchronize and a summary of each
   device allocation, surface object and texture object named in the packed
   argument buffer. D4R_CUDA_LAUNCH_DUMP_DIR additionally saves the raw
   contents of every summarized resource, one file per launch and argument. */
typedef struct AllocationRecord
{
    CUdeviceptr base;
    size_t bytes;
    struct AllocationRecord* next;
} AllocationRecord;

typedef struct SurfaceObjectRecord
{
    CUsurfObject object;
    CUarray array;
    struct SurfaceObjectRecord* next;
} SurfaceObjectRecord;

typedef struct FunctionNameRecord
{
    CUfunction function;
    char name[128];
    struct FunctionNameRecord* next;
} FunctionNameRecord;

/* CUDA_MEMCPY2D (v2 layout). */
typedef struct
{
    size_t srcXInBytes;
    size_t srcY;
    uint32_t srcMemoryType;
    uint32_t srcAlignment;
    const void* srcHost;
    CUdeviceptr srcDevice;
    CUarray srcArray;
    size_t srcPitch;
    size_t dstXInBytes;
    size_t dstY;
    uint32_t dstMemoryType;
    uint32_t dstAlignment;
    void* dstHost;
    CUdeviceptr dstDevice;
    CUarray dstArray;
    size_t dstPitch;
    size_t WidthInBytes;
    size_t Height;
} D4rMemcpy2D;

static pthread_mutex_t instrumentation_lock = PTHREAD_MUTEX_INITIALIZER;
static AllocationRecord* allocations;
static SurfaceObjectRecord* surface_objects;
static FunctionNameRecord* function_names;
static unsigned int launch_sequence;

static void remember_allocation(CUdeviceptr base, size_t bytes)
{
    AllocationRecord* record = (AllocationRecord*)malloc(sizeof(*record));
    if (record == NULL)
        return;
    record->base = base;
    record->bytes = bytes;
    pthread_mutex_lock(&instrumentation_lock);
    record->next = allocations;
    allocations = record;
    pthread_mutex_unlock(&instrumentation_lock);
}

static void forget_allocation(CUdeviceptr base)
{
    pthread_mutex_lock(&instrumentation_lock);
    for (AllocationRecord** current = &allocations; *current != NULL; current = &(*current)->next)
    {
        if ((*current)->base == base)
        {
            AllocationRecord* removed = *current;
            *current = removed->next;
            free(removed);
            break;
        }
    }
    pthread_mutex_unlock(&instrumentation_lock);
}

static int find_allocation(CUdeviceptr address, CUdeviceptr* base, size_t* bytes)
{
    int found = 0;
    pthread_mutex_lock(&instrumentation_lock);
    for (AllocationRecord* current = allocations; current != NULL; current = current->next)
    {
        if (address >= current->base && address < current->base + current->bytes)
        {
            *base = current->base;
            *bytes = current->bytes;
            found = 1;
            break;
        }
    }
    pthread_mutex_unlock(&instrumentation_lock);
    return found;
}

/* Explicit partial integration for game validation. Keep CUDA/ZLUDA handles
   visible to NGX; replace only the audited enc0 dispatch, using the same HIP
   context/default stream and linear pointers. This is NOT a ZLUDA-free backend. */
static pthread_once_t microcuda_k_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t microcuda_k_lock = PTHREAD_MUTEX_INITIALIZER;
static CUMODULELOADDATA_FN microcuda_k_load;
static CUMODULEGETFUNCTION_FN microcuda_k_get;
static CUMODULEUNLOAD_FN microcuda_k_unload;
static CULAUNCHKERNEL_FN microcuda_k_launch;
typedef void(__attribute__((sysv_abi)) *MICROCUDA_RANGE_FN)(uint64_t, size_t);
typedef void(__attribute__((sysv_abi)) *MICROCUDA_FREE_FN)(uint64_t);
typedef CUresult(__attribute__((sysv_abi)) *MICROCUDA_STATS_FN)(CUfunction, uint64_t*, uint64_t*);
static MICROCUDA_RANGE_FN microcuda_k_track, microcuda_k_write;
static MICROCUDA_FREE_FN microcuda_k_forget;
static MICROCUDA_STATS_FN microcuda_k_stats;
static CUmodule microcuda_k_reference[32], microcuda_k_modules[32];
static CUfunction microcuda_k_original[32], microcuda_k_functions[32];
static unsigned int microcuda_k_highwater;
static uint64_t microcuda_k_launches;

static void load_microcuda_k(void)
{
    const char* configured = getenv("D4R_MICROCUDA_LIBCUDA");
    if (configured == NULL || configured[0] == '\0')
    {
        tracef("microcuda-k unavailable: D4R_MICROCUDA_LIBCUDA required");
        return;
    }
    char path[1024];
    expand_home(configured, path, sizeof(path));
    void* library = load_library_resolving(path, 0);
    if (library == NULL) return;
    CUINIT_FN init = (CUINIT_FN)dlsym(library, "cuInit");
    microcuda_k_load = (CUMODULELOADDATA_FN)dlsym(library, "cuModuleLoadData");
    microcuda_k_get = (CUMODULEGETFUNCTION_FN)dlsym(library, "cuModuleGetFunction");
    microcuda_k_unload = (CUMODULEUNLOAD_FN)dlsym(library, "cuModuleUnload");
    microcuda_k_launch = (CULAUNCHKERNEL_FN)dlsym(library, "cuLaunchKernel");
    microcuda_k_track = (MICROCUDA_RANGE_FN)dlsym(library, "d4rMicrocudaTrackAllocation");
    microcuda_k_write = (MICROCUDA_RANGE_FN)dlsym(library, "d4rMicrocudaNotifyWrite");
    microcuda_k_forget = (MICROCUDA_FREE_FN)dlsym(library, "d4rMicrocudaForgetAllocation");
    microcuda_k_stats = (MICROCUDA_STATS_FN)dlsym(library, "d4rMicrocudaGetDispatchStats");
    if (microcuda_k_track)
    {
        pthread_mutex_lock(&instrumentation_lock);
        for (AllocationRecord* a = allocations; a != NULL; a = a->next)
            microcuda_k_track(a->base, a->bytes);
        pthread_mutex_unlock(&instrumentation_lock);
    }
    if (!init || !microcuda_k_load || !microcuda_k_get || !microcuda_k_unload || !microcuda_k_launch || init(0))
    {
        microcuda_k_load = NULL;
        tracef("microcuda-k initialization failed; no native MicroCUDA dispatch available");
    }
}

static void microcuda_k_note_module(CUmodule reference, const void* image)
{
    if (!microcuda_k_requested || native_source[0] == '\0') return;
    /* The experimental dispatch must use the same selected accuracy/fast/native
       set as the reference. A registry for another variant fails its SHA check. */
    setenv("D4R_MICROCUDA_NATIVE_DIR", native_source, 1);
    pthread_once(&microcuda_k_once, load_microcuda_k);
    if (!microcuda_k_load) return;
    CUmodule module = NULL;
    const CUresult result = microcuda_k_load(&module, image);
    if (result != CUDA_SUCCESS) return; /* explicit hybrid: unregistered modules use reference */
    pthread_mutex_lock(&microcuda_k_lock);
    unsigned int slot;
    for (slot = 0; slot < 32; ++slot) if (microcuda_k_modules[slot] == NULL) break;
    if (slot < 32)
    {
        microcuda_k_reference[slot] = reference;
        microcuda_k_modules[slot] = module;
        if (slot >= microcuda_k_highwater)
            __atomic_store_n(&microcuda_k_highwater, slot + 1, __ATOMIC_RELEASE);
        tracef("microcuda-k registered reference=%p native=%p slot=%u", reference, module, slot);
    }
    else
    {
        microcuda_k_unload(module);
        tracef("microcuda-k module table exhausted");
    }
    pthread_mutex_unlock(&microcuda_k_lock);
}

static void microcuda_k_note_function(CUmodule reference, CUfunction original, const char* name)
{
    if (!microcuda_k_requested || strcmp(name, "dltss_pwin_enc0_layer") != 0) return;
    pthread_mutex_lock(&microcuda_k_lock);
    int mapped = 0;
    for (unsigned int slot = 0; slot < microcuda_k_highwater; ++slot)
        if (microcuda_k_reference[slot] == reference && microcuda_k_modules[slot] != NULL)
        {
            CUfunction function = NULL;
            CUresult result = microcuda_k_get(&function, microcuda_k_modules[slot], name);
            if (result == CUDA_SUCCESS)
            {
                mapped = 1;
                __atomic_store_n(&microcuda_k_functions[slot], function, __ATOMIC_RELEASE);
                __atomic_store_n(&microcuda_k_original[slot], original, __ATOMIC_RELEASE);
                tracef("microcuda-k dispatch mapped %s original=%p native=%p", name, original, function);
            }
            else tracef("microcuda-k function lookup failed %d", result);
        }
    if (!mapped) tracef("microcuda-k enc0 remains reference: no compatible module/selected native variant");
    pthread_mutex_unlock(&microcuda_k_lock);
}

static CUfunction microcuda_k_lookup(CUfunction original)
{
    if (!microcuda_k_requested) return NULL;
    const unsigned int count = __atomic_load_n(&microcuda_k_highwater, __ATOMIC_ACQUIRE);
    for (unsigned int i = 0; i < count; ++i)
        if (__atomic_load_n(&microcuda_k_original[i], __ATOMIC_ACQUIRE) == original)
            return __atomic_load_n(&microcuda_k_functions[i], __ATOMIC_ACQUIRE);
    return NULL;
}

static void microcuda_k_remove_module(CUmodule reference)
{
    if (!microcuda_k_requested) return;
    pthread_mutex_lock(&microcuda_k_lock);
    for (unsigned int slot = 0; slot < microcuda_k_highwater; ++slot)
        if (microcuda_k_reference[slot] == reference && microcuda_k_modules[slot] != NULL)
        {
            if (microcuda_k_stats && microcuda_k_functions[slot])
            {
                uint64_t prep = 0, skips = 0;
                microcuda_k_stats(microcuda_k_functions[slot], &prep, &skips);
                tracef("microcuda-k prep stats runs=%llu skips=%llu", (unsigned long long)prep, (unsigned long long)skips);
            }
            __atomic_store_n(&microcuda_k_original[slot], NULL, __ATOMIC_RELEASE);
            __atomic_store_n(&microcuda_k_functions[slot], NULL, __ATOMIC_RELEASE);
            CUresult result = microcuda_k_unload(microcuda_k_modules[slot]);
            if (result != CUDA_SUCCESS) tracef("microcuda-k unload failed %d", result);
            microcuda_k_modules[slot] = NULL;
            microcuda_k_reference[slot] = NULL;
            tracef("microcuda-k completed dispatches=%llu", (unsigned long long)microcuda_k_launches);
        }
    pthread_mutex_unlock(&microcuda_k_lock);
}

static void remember_surface_object(CUsurfObject object, const void* resource)
{
    uint32_t resource_type = UINT32_MAX;
    CUarray array = NULL;
    memcpy(&resource_type, resource, sizeof(resource_type));
    memcpy(&array, (const unsigned char*)resource + 8, sizeof(array));
    if (resource_type != 0 /* CU_RESOURCE_TYPE_ARRAY */)
        return;
    SurfaceObjectRecord* record = (SurfaceObjectRecord*)malloc(sizeof(*record));
    if (record == NULL)
        return;
    record->object = object;
    record->array = array;
    pthread_mutex_lock(&instrumentation_lock);
    record->next = surface_objects;
    surface_objects = record;
    pthread_mutex_unlock(&instrumentation_lock);
}

/* d4r output redirect: surfaces on `redirect_array` carry {u32 'R2' | pitch / 8 @84, u64 pointer @88} in the zero
   padding after ZLUDA's CUDA-format word (d4r's native surface stores then write that linear memory). */
static CUarray redirect_array;
static CUdeviceptr redirect_pointer;
static uint32_t redirect_pitch;

static CUresult write_redirect_tail(CUsurfObject object, CUdeviceptr pointer, uint32_t pitch, int async)
{
    static unsigned char ring[256][16];
    static unsigned int next;
    unsigned char* tail = ring[next++ % 256];
    // dword 21: 0x5232 ('R2', RGBA16F texels) << 16 | pitch / 8 (0 = off); dwords 22-23: pointer.
    // Row pitches are multiples of 8, so bit 0 of `pitch` selects 'R3': R10G10B10A2_UNORM texels.
    const uint32_t word = pitch != 0 ? ((pitch & 1u) != 0 ? 0x52330000u : 0x52320000u) | ((pitch >> 3) & 0xffffu) : 0u;
    memset(tail, 0, 16);
    memcpy(tail, &word, 4);
    memcpy(tail + 4, &pointer, 8);
    CUMEMCPYHTODASYNC_FN function = (CUMEMCPYHTODASYNC_FN)find_zluda_symbol("cuMemcpyHtoDAsync_v2");
    if (function == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    CUresult result = function((CUdeviceptr)object + 84, tail, 12, NULL);
    if (result == CUDA_SUCCESS && !async)
    {
        CUCTX_SYNCHRONIZE_FN sync = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
        result = sync != NULL ? sync() : CUDA_ERROR_NOT_SUPPORTED;
    }
    return result;
}

static CUarray find_surface_array(CUsurfObject object)
{
    CUarray array = NULL;
    pthread_mutex_lock(&instrumentation_lock);
    for (SurfaceObjectRecord* current = surface_objects; current != NULL; current = current->next)
    {
        if (current->object == object)
        {
            array = current->array;
            break;
        }
    }
    pthread_mutex_unlock(&instrumentation_lock);
    return array;
}

static void remember_function_name(CUfunction function, const char* name)
{
    FunctionNameRecord* record = (FunctionNameRecord*)malloc(sizeof(*record));
    if (record == NULL || name == NULL)
    {
        free(record);
        return;
    }
    record->function = function;
    snprintf(record->name, sizeof(record->name), "%s", name);
    pthread_mutex_lock(&instrumentation_lock);
    record->next = function_names;
    function_names = record;
    pthread_mutex_unlock(&instrumentation_lock);
}

static const char* find_function_name(CUfunction function)
{
    const char* name = "unknown";
    pthread_mutex_lock(&instrumentation_lock);
    for (FunctionNameRecord* current = function_names; current != NULL; current = current->next)
    {
        if (current->function == function)
        {
            name = current->name;
            break;
        }
    }
    pthread_mutex_unlock(&instrumentation_lock);
    return name;
}

/* Opt-in per-kernel GPU elapsed time. Event pairs are recorded on the same
   stream as each launch, then read only after an existing context sync. This
   adds profiling overhead and must never be used as the FPS baseline. */
typedef struct PendingKernelProfile
{
    CUevent start;
    CUevent end;
    CUfunction function;
    unsigned int sequence;
    unsigned int eligible;
    struct PendingKernelProfile* next;
} PendingKernelProfile;

static pthread_once_t kernel_profile_once = PTHREAD_ONCE_INIT;
static pthread_mutex_t kernel_profile_lock = PTHREAD_MUTEX_INITIALIZER;
static PendingKernelProfile* kernel_profile_head;
static PendingKernelProfile* kernel_profile_tail;
static unsigned int kernel_profile_skip;
static unsigned int kernel_profile_limit;
static unsigned int kernel_profile_eligible;
static char kernel_profile_filter[128];
static int kernel_profile_enabled;
static CUEVENTCREATE_FN kernel_event_create;
static CUEVENTRECORD_FN kernel_event_record;
static CUEVENTELAPSED_FN kernel_event_elapsed;
static CUEVENTDESTROY_FN kernel_event_destroy;

static void configure_kernel_profile(void)
{
    const char* enabled = getenv("D4R_CUDA_KERNEL_PROFILE");
    kernel_profile_enabled = enabled != NULL && strcmp(enabled, "1") == 0;
    if (!kernel_profile_enabled)
        return;
    const char* skip = getenv("D4R_CUDA_KERNEL_PROFILE_SKIP");
    const char* limit = getenv("D4R_CUDA_KERNEL_PROFILE_LIMIT");
    kernel_profile_skip = skip != NULL ? (unsigned int)strtoul(skip, NULL, 0) : 0;
    kernel_profile_limit = limit != NULL ? (unsigned int)strtoul(limit, NULL, 0) : 3000;
    /* D4R_CUDA_KERNEL_PROFILE_FILTER restricts profiling to kernels whose name
       contains the given substring (e.g. "dltss"); SKIP and LIMIT then count
       only those launches, so a warmed gameplay window needs no knowledge of
       the global launch sequence. */
    const char* filter = getenv("D4R_CUDA_KERNEL_PROFILE_FILTER");
    if (filter != NULL)
        snprintf(kernel_profile_filter, sizeof(kernel_profile_filter), "%s", filter);
    kernel_event_create = (CUEVENTCREATE_FN)find_zluda_symbol("cuEventCreate");
    kernel_event_record = (CUEVENTRECORD_FN)find_zluda_symbol("cuEventRecord");
    kernel_event_elapsed = (CUEVENTELAPSED_FN)find_zluda_symbol("cuEventElapsedTime");
    kernel_event_destroy = (CUEVENTDESTROY_FN)find_zluda_symbol("cuEventDestroy_v2");
    if (kernel_event_create == NULL || kernel_event_record == NULL ||
        kernel_event_elapsed == NULL || kernel_event_destroy == NULL)
        kernel_profile_enabled = 0;
}

static void destroy_kernel_profile(PendingKernelProfile* profile)
{
    if (profile == NULL)
        return;
    if (profile->start != NULL)
        kernel_event_destroy(profile->start);
    if (profile->end != NULL)
        kernel_event_destroy(profile->end);
    free(profile);
}

static PendingKernelProfile* begin_kernel_profile(unsigned int sequence, CUfunction function, CUstream stream)
{
    pthread_once(&kernel_profile_once, configure_kernel_profile);
    if (!kernel_profile_enabled)
        return NULL;
    if (kernel_profile_filter[0] != '\0' &&
        strstr(find_function_name(function), kernel_profile_filter) == NULL)
        return NULL;
    pthread_mutex_lock(&kernel_profile_lock);
    const unsigned int eligible = ++kernel_profile_eligible;
    pthread_mutex_unlock(&kernel_profile_lock);
    if (eligible <= kernel_profile_skip || eligible - kernel_profile_skip > kernel_profile_limit)
        return NULL;
    PendingKernelProfile* profile = (PendingKernelProfile*)calloc(1, sizeof(*profile));
    if (profile == NULL)
        return NULL;
    profile->sequence = sequence;
    profile->eligible = eligible;
    profile->function = function;
    if (kernel_event_create(&profile->start, 0) != CUDA_SUCCESS ||
        kernel_event_create(&profile->end, 0) != CUDA_SUCCESS ||
        kernel_event_record(profile->start, stream) != CUDA_SUCCESS)
    {
        destroy_kernel_profile(profile);
        return NULL;
    }
    return profile;
}

static void end_kernel_profile(PendingKernelProfile* profile, CUresult launch_result, CUstream stream)
{
    if (profile == NULL)
        return;
    if (launch_result != CUDA_SUCCESS || kernel_event_record(profile->end, stream) != CUDA_SUCCESS)
    {
        destroy_kernel_profile(profile);
        return;
    }
    pthread_mutex_lock(&kernel_profile_lock);
    if (kernel_profile_tail != NULL)
        kernel_profile_tail->next = profile;
    else
        kernel_profile_head = profile;
    kernel_profile_tail = profile;
    pthread_mutex_unlock(&kernel_profile_lock);
}

static void flush_kernel_profiles(void)
{
    pthread_once(&kernel_profile_once, configure_kernel_profile);
    if (!kernel_profile_enabled)
        return;
    pthread_mutex_lock(&kernel_profile_lock);
    PendingKernelProfile* profile = kernel_profile_head;
    kernel_profile_head = kernel_profile_tail = NULL;
    pthread_mutex_unlock(&kernel_profile_lock);
    while (profile != NULL)
    {
        PendingKernelProfile* next = profile->next;
        float elapsed = -1.0f;
        if (kernel_event_elapsed(&elapsed, profile->start, profile->end) == CUDA_SUCCESS)
            tracef("KERNEL_PROFILE launch=%u eligible=%u gpu_ms=%.6f name=%s", profile->sequence,
                   profile->eligible, elapsed, find_function_name(profile->function));
        destroy_kernel_profile(profile);
        profile = next;
    }
}

static float half_bits_to_float(uint16_t bits)
{
    const uint32_t sign = (uint32_t)(bits & 0x8000u) << 16;
    const uint32_t exponent = (bits >> 10) & 0x1fu;
    const uint32_t mantissa = bits & 0x3ffu;
    uint32_t result;
    if (exponent == 0)
    {
        float magnitude = (float)mantissa * (1.0f / 16777216.0f);
        memcpy(&result, &magnitude, sizeof(result));
        result |= sign;
    }
    else if (exponent == 0x1f)
        result = sign | 0x7f800000u | (mantissa << 13);
    else
        result = sign | ((exponent + 112u) << 23) | (mantissa << 13);
    float value;
    memcpy(&value, &result, sizeof(value));
    return value;
}

/* Bytes per element of the backing storage ZLUDA creates for a CUDA array
   format (see zluda/src/impl/array.rs for the remapped formats). */
static size_t array_element_bytes(int32_t format, uint32_t channels)
{
    switch (format)
    {
    case 0x01: case 0x08: return 1u * channels;
    case 0x02: case 0x09: case 0x10: return 2u * channels;
    case 0x03: case 0x0a: case 0x20: return 4u * channels;
    case 80: return 4; /* CUDA copies packed 10:10:10:2 even though ZLUDA's backing is RGBA32F */
    case 192: case 198: return 1;
    case 193: case 199: return 2;
    case 194: case 200: return 4;
    case 195: case 201: return 2;
    case 196: case 202: return 4;
    case 197: case 203: return 8;
    default: return 0;
    }
}

/* Summarizes a resource: nonzero bytes, a content hash, and value ranges
   when read as half or single floats (the element type is often unknown). */
static void summarize_resource(unsigned int sequence, const char* kernel, size_t argument_offset,
                               const char* kind, const char* shape, int32_t format,
                               const unsigned char* data, size_t bytes)
{
    size_t nonzero_bytes = 0;
    uint64_t hash = 1469598103934665603ull;
    for (size_t index = 0; index < bytes; ++index)
    {
        nonzero_bytes += data[index] != 0;
        hash = (hash ^ data[index]) * 1099511628211ull;
    }
    size_t half_nan = 0, half_nonzero = 0;
    float half_min = 0.0f, half_max = 0.0f;
    double half_abs_sum = 0.0;
    for (size_t index = 0; index + 1 < bytes; index += 2)
    {
        uint16_t bits;
        memcpy(&bits, data + index, sizeof(bits));
        const float value = half_bits_to_float(bits);
        if (value != value || (bits & 0x7c00u) == 0x7c00u)
        {
            ++half_nan;
            continue;
        }
        if (value != 0.0f)
            ++half_nonzero;
        half_min = value < half_min ? value : half_min;
        half_max = value > half_max ? value : half_max;
        half_abs_sum += value < 0.0f ? -value : value;
    }
    size_t float_nan = 0;
    float float_min = 0.0f, float_max = 0.0f;
    for (size_t index = 0; index + 3 < bytes; index += 4)
    {
        float value;
        memcpy(&value, data + index, sizeof(value));
        uint32_t bits;
        memcpy(&bits, &value, sizeof(bits));
        if ((bits & 0x7f800000u) == 0x7f800000u)
        {
            ++float_nan;
            continue;
        }
        float_min = value < float_min ? value : float_min;
        float_max = value > float_max ? value : float_max;
    }
    tracef("launch[%u] %s arg[%zu] %s %s fmt=%d bytes=%zu nonzero_bytes=%zu hash=%016llx "
           "f16{nonzero=%zu nan_inf=%zu min=%g max=%g mean_abs=%g} f32{nan_inf=%zu min=%g max=%g}",
           sequence, kernel, argument_offset, kind, shape, format, bytes, nonzero_bytes,
           (unsigned long long)hash, half_nonzero, half_nan, half_min, half_max,
           bytes >= 2 ? half_abs_sum / (double)(bytes / 2) : 0.0, float_nan, float_min, float_max);

    char dump_dir[MAX_PATH] = {0};
    DWORD length = GetEnvironmentVariableA("D4R_CUDA_LAUNCH_DUMP_DIR", dump_dir, sizeof(dump_dir));
    if (length == 0 || length >= sizeof(dump_dir))
        return;
    mkdir(dump_dir, 0700);
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/launch-%03u-%s-arg%03zu-%s-%s-fmt%d.bin", dump_dir,
             sequence, kernel, argument_offset, kind, shape, format);
    FILE* output = fopen(filename, "wb");
    if (output != NULL)
    {
        fwrite(data, 1, bytes, output);
        fclose(output);
    }
}

static int read_array(CUarray array, unsigned char** data, size_t* bytes, char* shape,
                      size_t shape_size, int32_t* format)
{
    CUDA_ARRAY_DESCRIPTOR_V2 descriptor;
    if (!lookup_array_descriptor(array, &descriptor))
        return 0;
    const size_t element_bytes = array_element_bytes(descriptor.Format, descriptor.NumChannels);
    if (element_bytes == 0)
        return 0;
    CUMEMCPY2D_FN copy = (CUMEMCPY2D_FN)find_zluda_symbol("cuMemcpy2D_v2");
    const size_t height = descriptor.Height != 0 ? descriptor.Height : 1;
    const size_t row_bytes = descriptor.Width * element_bytes;
    unsigned char* buffer = (unsigned char*)malloc(row_bytes * height);
    if (copy == NULL || buffer == NULL)
    {
        free(buffer);
        return 0;
    }
    D4rMemcpy2D request;
    memset(&request, 0, sizeof(request));
    request.srcMemoryType = 3; /* CU_MEMORYTYPE_ARRAY */
    request.srcArray = array;
    request.dstMemoryType = 1; /* CU_MEMORYTYPE_HOST */
    request.dstHost = buffer;
    request.dstPitch = row_bytes;
    request.WidthInBytes = row_bytes;
    request.Height = height;
    const CUresult result = copy(&request);
    if (result != CUDA_SUCCESS)
    {
        tracef("launch stats: array %p readback failed with %d", array, result);
        free(buffer);
        return 0;
    }
    snprintf(shape, shape_size, "%zux%zux%u", descriptor.Width, height, descriptor.NumChannels);
    *format = descriptor.Format;
    *data = buffer;
    *bytes = row_bytes * height;
    return 1;
}

static void summarize_launch(unsigned int sequence, CUfunction function_handle, void** extra)
{
    if (extra == NULL || (uintptr_t)extra[0] != 1 || extra[1] == NULL ||
        (uintptr_t)extra[2] != 2 || extra[3] == NULL)
        return;
    CUCTX_SYNCHRONIZE_FN synchronize = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
    CUMEMCPYDTOH_FN copy_to_host = (CUMEMCPYDTOH_FN)find_zluda_symbol("cuMemcpyDtoH_v2");
    if (synchronize == NULL || copy_to_host == NULL)
        return;
    const CUresult sync_result = synchronize();
    const char* kernel = find_function_name(function_handle);
    tracef("launch[%u] %s synchronize result=%d", sequence, kernel, sync_result);
    const unsigned char* arguments = (const unsigned char*)extra[1];
    const size_t argument_bytes = *(const size_t*)extra[3];
    CUdeviceptr seen[64];
    size_t seen_count = 0;
    for (size_t offset = 0; offset + 8 <= argument_bytes; offset += 8)
    {
        uint64_t word;
        memcpy(&word, arguments + offset, sizeof(word));
        if (word == 0)
            continue;
        unsigned char* data = NULL;
        size_t bytes = 0;
        char shape[64] = "linear";
        int32_t format = -1;
        const char* kind = NULL;
        CUarray array = find_surface_array((CUsurfObject)word);
        unsigned char descriptor[CUDA_RESOURCE_DESC_BYTES];
        if (array != NULL)
            kind = "surface";
        else if (lookup_texture_descriptor((CUtexObject)word, descriptor))
        {
            uint32_t resource_type;
            memcpy(&resource_type, descriptor, sizeof(resource_type));
            if (resource_type == 0)
            {
                memcpy(&array, descriptor + 8, sizeof(array));
                kind = "texture";
            }
            else
                memcpy(&word, descriptor + 8, sizeof(word)); /* linear/pitch2D devPtr */
        }
        if (kind != NULL)
        {
            if (!read_array(array, &data, &bytes, shape, sizeof(shape), &format))
                continue;
        }
        else
        {
            CUdeviceptr base = 0;
            size_t allocation_bytes = 0;
            if (!find_allocation((CUdeviceptr)word, &base, &allocation_bytes))
                continue;
            int duplicate = 0;
            for (size_t index = 0; index < seen_count; ++index)
                duplicate |= seen[index] == base;
            if (duplicate)
                continue;
            if (seen_count < sizeof(seen) / sizeof(seen[0]))
                seen[seen_count++] = base;
            data = (unsigned char*)malloc(allocation_bytes);
            if (data == NULL || copy_to_host(data, base, allocation_bytes) != CUDA_SUCCESS)
            {
                free(data);
                continue;
            }
            bytes = allocation_bytes;
            kind = "buffer";
            snprintf(shape, sizeof(shape), "base0x%llx+0x%llx", (unsigned long long)base,
                     (unsigned long long)((CUdeviceptr)word - base));
        }
        summarize_resource(sequence, kernel, offset, kind, shape, format, data, bytes);
        free(data);
    }
}

/* Opt-in replay capture. With D4R_CUDA_REPLAY_DUMP_DIR set, launches whose
   kernel name contains D4R_CUDA_REPLAY_DUMP_FILTER (default: all) are
   captured after D4R_CUDA_REPLAY_DUMP_SKIP matching launches, up to
   D4R_CUDA_REPLAY_DUMP_LIMIT (default 1). Each capture synchronizes and then
   writes, before the launch runs, the raw packed argument buffer and every
   device allocation it points into, with a manifest that tools/
   bench_dlss_kernel.cpp uses to replay the launch in isolation. */
static unsigned int replay_dump_matches;

static void replay_dump_launch(unsigned int sequence, CUfunction function_handle, unsigned int grid_x,
                               unsigned int grid_y, unsigned int grid_z, unsigned int block_x,
                               unsigned int block_y, unsigned int block_z, unsigned int shared_bytes,
                               void** extra)
{
    static int configured;
    static char directory[512];
    static char filter[128];
    static unsigned int skip, limit;
    if (!configured)
    {
        const char* value = getenv("D4R_CUDA_REPLAY_DUMP_DIR");
        snprintf(directory, sizeof(directory), "%s", value != NULL ? value : "");
        value = getenv("D4R_CUDA_REPLAY_DUMP_FILTER");
        snprintf(filter, sizeof(filter), "%s", value != NULL ? value : "");
        value = getenv("D4R_CUDA_REPLAY_DUMP_SKIP");
        skip = value != NULL ? (unsigned int)strtoul(value, NULL, 0) : 0;
        value = getenv("D4R_CUDA_REPLAY_DUMP_LIMIT");
        limit = value != NULL ? (unsigned int)strtoul(value, NULL, 0) : 1;
        configured = 1;
    }
    if (directory[0] == '\0' || extra == NULL || (uintptr_t)extra[0] != 1 || extra[1] == NULL ||
        (uintptr_t)extra[2] != 2 || extra[3] == NULL)
        return;
    const char* kernel = find_function_name(function_handle);
    if (filter[0] != '\0' && strstr(kernel, filter) == NULL)
        return;
    const unsigned int match = ++replay_dump_matches;
    if (match <= skip || match - skip > limit)
        return;
    CUCTX_SYNCHRONIZE_FN synchronize = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
    CUMEMCPYDTOH_FN copy_to_host = (CUMEMCPYDTOH_FN)find_zluda_symbol("cuMemcpyDtoH_v2");
    if (synchronize == NULL || copy_to_host == NULL || synchronize() != CUDA_SUCCESS)
        return;

    char path[1024];
    snprintf(path, sizeof(path), "%s/replay-%06u-%s", directory, sequence, kernel);
    mkdir(directory, 0700);
    mkdir(path, 0700);
    const unsigned char* arguments = (const unsigned char*)extra[1];
    const size_t argument_bytes = *(const size_t*)extra[3];
    char filename[1200];
    snprintf(filename, sizeof(filename), "%s/args.bin", path);
    FILE* output = fopen(filename, "wb");
    if (output == NULL)
        return;
    fwrite(arguments, 1, argument_bytes, output);
    fclose(output);
    snprintf(filename, sizeof(filename), "%s/manifest.txt", path);
    FILE* manifest = fopen(filename, "w");
    if (manifest == NULL)
        return;
    fprintf(manifest, "kernel %s\nlaunch %u %u %u %u %u %u %u\nargs %zu\n", kernel, grid_x, grid_y, grid_z,
            block_x, block_y, block_z, shared_bytes, argument_bytes);
    CUdeviceptr seen[64];
    size_t seen_count = 0;
    for (size_t offset = 0; offset + 8 <= argument_bytes; offset += 8)
    {
        uint64_t word;
        memcpy(&word, arguments + offset, sizeof(word));
        CUdeviceptr base = 0;
        size_t bytes = 0;
        if (word == 0)
            continue;
        if (find_surface_array((CUsurfObject)word) != NULL)
        {
            fprintf(manifest, "surface %zu\n", offset);
            continue;
        }
        unsigned char descriptor[CUDA_RESOURCE_DESC_BYTES];
        if (lookup_texture_descriptor((CUtexObject)word, descriptor))
        {
            fprintf(manifest, "texture %zu\n", offset);
            continue;
        }
        if (!find_allocation((CUdeviceptr)word, &base, &bytes))
            continue;
        size_t index = 0;
        while (index < seen_count && seen[index] != base)
            ++index;
        if (index == seen_count && seen_count < sizeof(seen) / sizeof(seen[0]))
        {
            seen[seen_count++] = base;
            unsigned char* data = (unsigned char*)malloc(bytes);
            if (data != NULL && copy_to_host(data, base, bytes) == CUDA_SUCCESS)
            {
                snprintf(filename, sizeof(filename), "%s/alloc-%zu.bin", path, index);
                FILE* allocation = fopen(filename, "wb");
                if (allocation != NULL)
                {
                    fwrite(data, 1, bytes, allocation);
                    fclose(allocation);
                }
                fprintf(manifest, "alloc %zu 0x%llx %zu\n", index, (unsigned long long)base, bytes);
            }
            free(data);
        }
        if (index < seen_count)
            fprintf(manifest, "pointer %zu %zu %llu\n", offset, index, (unsigned long long)(word - base));
    }
    fclose(manifest);
    tracef("replay capture launch[%u] %s -> %s", sequence, kernel, path);
}

static int launch_stats_enabled(void)
{
    char value[8] = {0};
    DWORD length = GetEnvironmentVariableA("D4R_CUDA_LAUNCH_STATS", value, sizeof(value));
    return length > 0 && length < sizeof(value) && value[0] == '1';
}

static size_t readable_span(const void* address)
{
    FILE* maps = fopen("/proc/self/maps", "r");
    if (maps == NULL)
        return 0;
    const uintptr_t target = (uintptr_t)address;
    char line[512];
    size_t span = 0;
    while (fgets(line, sizeof(line), maps) != NULL)
    {
        unsigned long long begin = 0, end = 0;
        char permissions[5] = {0};
        if (sscanf(line, "%llx-%llx %4s", &begin, &end, permissions) == 3 &&
            permissions[0] == 'r' && target >= begin && target < end)
        {
            span = (size_t)(end - target);
            break;
        }
    }
    fclose(maps);
    return span;
}

static void capture_ptx_if_present(const void* image, const char* api)
{
    /* D4R_CUDA_CAPTURE=0 disables module capture (e.g. for game sessions). */
    char capture[8] = {0};
    DWORD capture_length = GetEnvironmentVariableA("D4R_CUDA_CAPTURE", capture, sizeof(capture));
    if (image == NULL || (capture_length > 0 && capture_length < sizeof(capture) && capture[0] == '0'))
        return;
    const size_t span = readable_span(image);
    if (span == 0)
    {
        tracef("%s module image=%p is not in a readable mapping", api, image);
        return;
    }

    const unsigned char* bytes = (const unsigned char*)image;

    /* CUDA fatbin v1 header: magic, version, header size, payload size. */
    uint32_t fatbin_magic = 0;
    memcpy(&fatbin_magic, bytes, sizeof(fatbin_magic));
    if (fatbin_magic == 0xba55ed50u && span >= 16)
    {
        uint16_t version = 0;
        uint16_t header_size = 0;
        uint64_t files_size = 0;
        memcpy(&version, bytes + 4, sizeof(version));
        memcpy(&header_size, bytes + 6, sizeof(header_size));
        memcpy(&files_size, bytes + 8, sizeof(files_size));
        if (version == 1 && header_size >= 16 && header_size <= 4096 &&
            files_size <= 64u * 1024u * 1024u && header_size + files_size <= span)
        {
            const size_t module_size = (size_t)header_size + (size_t)files_size;
            char capture_dir[MAX_PATH] = {0};
            DWORD length = GetEnvironmentVariableA("D4R_CUDA_CAPTURE_DIR", capture_dir, sizeof(capture_dir));
            if (length == 0 || length >= sizeof(capture_dir))
                strcpy(capture_dir, "/tmp/d4r-dlss-cuda-modules");
            mkdir(capture_dir, 0700);

            pthread_mutex_lock(&trace_lock);
            const unsigned int sequence = ++capture_sequence;
            pthread_mutex_unlock(&trace_lock);
            char filename[MAX_PATH];
            snprintf(filename, sizeof(filename), "%s/dlss-module-%ld-%04u.fatbin",
                     capture_dir, (long)getpid(), sequence);
            FILE* output = fopen(filename, "wb");
            if (output == NULL)
                tracef("%s saw CUDA fatbin v1 size=%zu but could not open %s", api, module_size, filename);
            else
            {
                const size_t written = fwrite(bytes, 1, module_size, output);
                fclose(output);
                tracef("%s captured CUDA fatbin v1 size=%zu (header=%u payload=%llu) to %s (%s)",
                       api, module_size, header_size, (unsigned long long)files_size, filename,
                       written == module_size ? "complete" : "short write");
            }
            return;
        }
        tracef("%s found CUDA fatbin magic with implausible v1 header (version=%u header=%u files=%llu span=%zu)",
               api, version, header_size, (unsigned long long)files_size, span);
        return;
    }

    const size_t scan_limit = span < 256 ? span : 256;
    size_t ptx_offset = SIZE_MAX;
    for (size_t i = 0; i + 8 <= scan_limit; ++i)
    {
        if (memcmp(bytes + i, ".version", 8) == 0)
        {
            ptx_offset = i;
            break;
        }
    }
    if (ptx_offset == SIZE_MAX)
    {
        char prefix[64] = {0};
        size_t prefix_size = span < 24 ? span : 24;
        for (size_t i = 0; i < prefix_size; ++i)
            snprintf(prefix + i * 2, sizeof(prefix) - i * 2, "%02x", bytes[i]);
        prefix[prefix_size * 2] = '\0';
        tracef("%s module image=%p readable=%zu bytes; no PTX .version in prefix (hex=%s)",
               api, image, span, prefix);
        return;
    }

    size_t maximum = span - ptx_offset;
    if (maximum > 64u * 1024u * 1024u)
        maximum = 64u * 1024u * 1024u;
    const unsigned char* ptx = bytes + ptx_offset;
    const unsigned char* terminator = (const unsigned char*)memchr(ptx, 0, maximum);
    if (terminator == NULL)
    {
        tracef("%s found PTX at image+%zu but no terminator within %zu readable bytes", api, ptx_offset, maximum);
        return;
    }
    const size_t ptx_size = (size_t)(terminator - ptx);

    char capture_dir[MAX_PATH] = {0};
    DWORD length = GetEnvironmentVariableA("D4R_CUDA_CAPTURE_DIR", capture_dir, sizeof(capture_dir));
    if (length == 0 || length >= sizeof(capture_dir))
        strcpy(capture_dir, "/tmp/d4r-dlss-cuda-modules");
    mkdir(capture_dir, 0700);

    pthread_mutex_lock(&trace_lock);
    const unsigned int sequence = ++capture_sequence;
    pthread_mutex_unlock(&trace_lock);
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/dlss-module-%ld-%04u.ptx", capture_dir, (long)getpid(), sequence);
    FILE* output = fopen(filename, "wb");
    if (output == NULL)
    {
        tracef("%s found PTX size=%zu at image+%zu but capture could not be opened: %s",
               api, ptx_size, ptx_offset, filename);
        return;
    }
    const size_t written = fwrite(ptx, 1, ptx_size, output);
    fclose(output);
    tracef("%s captured PTX size=%zu image+%zu to %s (%s)", api, ptx_size, ptx_offset,
           filename, written == ptx_size ? "complete" : "short write");
}

static CUresult missing(const char* name)
{
    tracef("ZLUDA does not export %s", name);
    return CUDA_ERROR_NOT_SUPPORTED;
}

static void* bridge_export(const char* name)
{
    HMODULE module = GetModuleHandleA("nvcuda.dll");
    return module != NULL ? (void*)GetProcAddress(module, name) : NULL;
}

/* Process exit. NVIDIA's NGX DLL unloads its CUDA modules from its own detach, which runs before the
   NGX shim's (it was loaded later). A Vulkan game can exit with a command buffer still waiting inside
   the GPU ring for the shim's output event; that teardown then blocks on the stuck job until the ring
   times out and resets. The shim registers a hook releasing those waits; it runs once, before the
   first teardown call made while Wine is shutting the process down. */
static void (WINAPI* shutdown_hook)(void);

void WINAPI d4rSetShutdownHook(void (WINAPI* hook)(void))
{
    shutdown_hook = hook;
}

/* Returns 1 while Wine shuts the process down. ExitProcess has killed the other threads by then,
   possibly inside HIP/HSA with a runtime lock held, so a teardown call into HIP can block forever
   (seen: NVIDIA's NGX Shutdown1 -> cuTexObjectDestroy -> HSA mutex). Teardown and sync calls then
   report success without touching HIP; the kernel reclaims the process's GPU state. */
static int before_teardown(void)
{
    static BOOLEAN (WINAPI* in_progress)(void);
    static LONG ran;
    if (in_progress == NULL)
        in_progress = (BOOLEAN (WINAPI*)(void))GetProcAddress(GetModuleHandleA("ntdll.dll"), "RtlDllShutdownInProgress");
    if (in_progress == NULL || !in_progress())
        return 0;
    if (shutdown_hook != NULL && InterlockedExchange(&ran, 1) == 0)
    {
        tracef("process shutdown: running the shim's hook; CUDA teardown is skipped");
        shutdown_hook();
    }
    return 1;
}

CUresult WINAPI cuInit(unsigned int flags)
{
    CUINIT_FN function = (CUINIT_FN)find_zluda_symbol("cuInit");
    CUresult result = function != NULL ? function(flags) : CUDA_ERROR_NOT_INITIALIZED;
    tracef("cuInit flags=%u result=%d", flags, result);
    return result;
}

CUresult WINAPI cuDeviceGetCount(int* count)
{
    CUDEVICEGETCOUNT_FN function = (CUDEVICEGETCOUNT_FN)find_zluda_symbol("cuDeviceGetCount");
    CUresult result = function != NULL ? function(count) : CUDA_ERROR_NOT_INITIALIZED;
    tracef("cuDeviceGetCount result=%d count=%d", result, count != NULL ? *count : -1);
    return result;
}

CUresult WINAPI cuDeviceGet(CUdevice* device, int ordinal)
{
    CUDEVICEGET_FN function = (CUDEVICEGET_FN)find_zluda_symbol("cuDeviceGet");
    CUresult result = function != NULL ? function(device, ordinal) : CUDA_ERROR_NOT_INITIALIZED;
    tracef("cuDeviceGet ordinal=%d result=%d device=%d", ordinal, result, device != NULL ? *device : -1);
    return result;
}

CUresult WINAPI cuCtxPushCurrent_v2(CUcontext context)
{
    CUCTX_CURRENT_FN function = (CUCTX_CURRENT_FN)find_zluda_symbol("cuCtxPushCurrent_v2");
    CUresult result = function != NULL ? function(context) : CUDA_ERROR_NOT_INITIALIZED;
    TRACE_CALL(result, "cuCtxPushCurrent_v2 context=%p result=%d", context, result);
    return result;
}

CUresult WINAPI cuCtxPopCurrent_v2(CUcontext* context)
{
    CUCTX_POP_CURRENT_FN function = (CUCTX_POP_CURRENT_FN)find_zluda_symbol("cuCtxPopCurrent_v2");
    CUresult result = function != NULL ? function(context) : CUDA_ERROR_NOT_INITIALIZED;
    TRACE_CALL(result, "cuCtxPopCurrent_v2 result=%d context=%p", result, context != NULL ? *context : NULL);
    return result;
}

CUresult WINAPI cuCtxGetDevice(CUdevice* device)
{
    CUCTX_GET_DEVICE_FN function = (CUCTX_GET_DEVICE_FN)find_zluda_symbol("cuCtxGetDevice");
    if (function == NULL)
        return CUDA_ERROR_NOT_INITIALIZED;
    ensure_context();
    CUresult result = context_setup_result == CUDA_SUCCESS ? function(device) : context_setup_result;
    TRACE_CALL(result, "cuCtxGetDevice result=%d device=%d", result, device != NULL ? *device : -1);
    return result;
}

CUresult WINAPI cuCtxCreate_v2(CUcontext* context, unsigned int flags, CUdevice device)
{
    CUCTXCREATE_FN function = (CUCTXCREATE_FN)find_zluda_symbol("cuCtxCreate_v2");
    CUresult result = function != NULL ? function(context, flags, device) : CUDA_ERROR_NOT_INITIALIZED;
    tracef("cuCtxCreate_v2 flags=0x%x device=%d result=%d context=%p", flags, device,
           result, context != NULL ? *context : NULL);
    return result;
}

CUresult WINAPI cuCtxDestroy_v2(CUcontext context)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUCTXDESTROY_FN function = (CUCTXDESTROY_FN)find_zluda_symbol("cuCtxDestroy_v2");
    CUresult result = function != NULL ? function(context) : CUDA_ERROR_NOT_INITIALIZED;
    tracef("cuCtxDestroy_v2 context=%p result=%d", context, result);
    return result;
}

CUresult WINAPI cuCtxSetCurrent(CUcontext context)
{
    CUCTX_CURRENT_FN function = (CUCTX_CURRENT_FN)find_zluda_symbol("cuCtxSetCurrent");
    CUresult result = function != NULL ? function(context) : missing("cuCtxSetCurrent");
    TRACE_CALL(result, "cuCtxSetCurrent context=%p result=%d", context, result);
    return result;
}

/* d4r: D4R_ELIDE_NGX_SYNC=1 turns the application's (NGX's) context and event synchronisations into
   no-ops. NGX issues everything on the null stream, so its GPU work stays ordered; the waits only
   stall the GPU while the CPU wakes up. The d4r shim synchronises through d4rCtxSynchronize. */
static int elide_ngx_sync(void)
{
    static int elide = -1;
    if (elide < 0)
    {
        const char* value = getenv("D4R_ELIDE_NGX_SYNC");
        elide = value != NULL && value[0] == '1';
    }
    return elide;
}

CUresult WINAPI d4rCtxSynchronize(void)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUCTX_SYNCHRONIZE_FN function = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
    CUresult result = function != NULL ? function() : missing("cuCtxSynchronize");
    if (result == CUDA_SUCCESS)
        flush_kernel_profiles();
    TRACE_CALL(result, "d4rCtxSynchronize result=%d", result);
    return result;
}

CUresult WINAPI cuCtxSynchronize(void)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    if (elide_ngx_sync())
    {
        TRACE_CALL(CUDA_SUCCESS, "cuCtxSynchronize elided");
        return CUDA_SUCCESS;
    }
    CUCTX_SYNCHRONIZE_FN function = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
    CUresult result = function != NULL ? function() : missing("cuCtxSynchronize");
    if (result == CUDA_SUCCESS)
        flush_kernel_profiles();
    TRACE_CALL(result, "cuCtxSynchronize result=%d", result);
    return result;
}

CUresult WINAPI cuDeviceGetLuid(char* luid, unsigned int* device_node_mask, CUdevice device)
{
    CUDEVICEGETLUID_FN function = (CUDEVICEGETLUID_FN)find_zluda_symbol("cuDeviceGetLuid");
    if (function == NULL)
        return missing("cuDeviceGetLuid");
    CUresult result = function(luid, device_node_mask, device);
    if (result == CUDA_SUCCESS && luid != NULL && device_node_mask != NULL)
    {
        unsigned int low = get_process_u32("D4R_CUDA_LUID_LOW", 0xffffffffu);
        unsigned int high = get_process_u32("D4R_CUDA_LUID_HIGH", 0xffffffffu);
        unsigned int node_mask = get_process_u32("D4R_CUDA_NODE_MASK", 0xffffffffu);
        if (low != 0xffffffffu)
            memcpy(luid, &low, sizeof(low));
        if (high != 0xffffffffu)
            memcpy(luid + sizeof(low), &high, sizeof(high));
        if (node_mask != 0xffffffffu)
            *device_node_mask = node_mask;
    }
    tracef("cuDeviceGetLuid dev=%d luid=%02x%02x%02x%02x%02x%02x%02x%02x nodeMask=%u result=%d",
           device, luid != NULL ? (unsigned char)luid[0] : 0,
           luid != NULL ? (unsigned char)luid[1] : 0, luid != NULL ? (unsigned char)luid[2] : 0,
           luid != NULL ? (unsigned char)luid[3] : 0, luid != NULL ? (unsigned char)luid[4] : 0,
           luid != NULL ? (unsigned char)luid[5] : 0, luid != NULL ? (unsigned char)luid[6] : 0,
           luid != NULL ? (unsigned char)luid[7] : 0,
           device_node_mask != NULL ? *device_node_mask : 0, result);
    return result;
}

CUresult WINAPI cuDeviceGetAttribute(int* value, int attribute, CUdevice device)
{
    CUDEVICEGETATTRIBUTE_FN function = (CUDEVICEGETATTRIBUTE_FN)find_zluda_symbol("cuDeviceGetAttribute");
    CUresult result = function != NULL ? function(value, attribute, device) : missing("cuDeviceGetAttribute");
    TRACE_CALL(result, "cuDeviceGetAttribute attribute=%d dev=%d result=%d value=%d",
           attribute, device, result, value != NULL ? *value : -1);
    return result;
}

CUresult WINAPI cuDeviceGetUuid(void* uuid, CUdevice device)
{
    CUDEVICEGETUUID_FN function = (CUDEVICEGETUUID_FN)find_zluda_symbol("cuDeviceGetUuid");
    CUresult result = function != NULL ? function(uuid, device) : missing("cuDeviceGetUuid");
    tracef("cuDeviceGetUuid device=%d result=%d uuid_ptr=%p", device, result, uuid);
    return result;
}

CUresult WINAPI cuGetErrorString(CUresult code, const char** message)
{
    CUGETERRORSTRING_FN function = (CUGETERRORSTRING_FN)find_zluda_symbol("cuGetErrorString");
    CUresult result = function != NULL ? function(code, message) : missing("cuGetErrorString");
    tracef("cuGetErrorString code=%d result=%d message=%s", code, result,
           message != NULL && *message != NULL ? *message : "<null>");
    return result;
}

/* D4R_CUDA_REPLACE_DIR: a debugging hook that loads <dir>/<fnv1a64>.ptx in place of
 * a CUDA fatbin v1 image with that FNV-1a hash (e.g. an instrumented kernel). */
static const void* replacement_image(const void* image, const char* api)
{
    char directory[MAX_PATH] = {0};
    DWORD length = GetEnvironmentVariableA("D4R_CUDA_REPLACE_DIR", directory, sizeof(directory));
    if (image == NULL || length == 0 || length >= sizeof(directory) || readable_span(image) < 16)
        return image;
    const unsigned char* bytes = (const unsigned char*)image;
    uint32_t magic = 0;
    uint16_t header_size = 0;
    uint64_t files_size = 0;
    memcpy(&magic, bytes, sizeof(magic));
    memcpy(&header_size, bytes + 6, sizeof(header_size));
    memcpy(&files_size, bytes + 8, sizeof(files_size));
    if (magic != 0xba55ed50u || header_size < 16 || files_size > 64u * 1024u * 1024u ||
        header_size + files_size > readable_span(image))
        return image;
    uint64_t hash = 0xcbf29ce484222325ull;
    for (size_t i = 0; i < (size_t)header_size + (size_t)files_size; ++i)
        hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    char filename[MAX_PATH];
    snprintf(filename, sizeof(filename), "%s/%016llx.ptx", directory, (unsigned long long)hash);
    FILE* input = fopen(filename, "rb");
    if (input == NULL)
        return image;
    fseek(input, 0, SEEK_END);
    const long size = ftell(input);
    fseek(input, 0, SEEK_SET);
    char* text = size > 0 ? (char*)malloc((size_t)size + 1) : NULL;
    if (text == NULL || fread(text, 1, (size_t)size, input) != (size_t)size)
    {
        fclose(input);
        free(text);
        return image;
    }
    fclose(input);
    text[size] = '\0';
    tracef("%s replaced fatbin %016llx with %s", api, (unsigned long long)hash, filename);
    return text; /* kept alive: modules may reference it */
}

CUresult WINAPI cuModuleLoadData(CUmodule* module, const void* image)
{
    CUMODULELOADDATA_FN function = (CUMODULELOADDATA_FN)find_zluda_symbol("cuModuleLoadData");
    verify_native_kernels(image);
    capture_ptx_if_present(image, "cuModuleLoadData");
    image = replacement_image(image, "cuModuleLoadData");
    CUresult result = function != NULL ? function(module, image) : missing("cuModuleLoadData");
    if (result == CUDA_SUCCESS) microcuda_k_note_module(*module, image);
    tracef("cuModuleLoadData image=%p result=%d module=%p", image, result, module != NULL ? *module : NULL);
    return result;
}

CUresult WINAPI cuModuleLoadDataEx(CUmodule* module, const void* image, unsigned int option_count,
                                   int* options, void** option_values)
{
    CUMODULELOADDATAEX_FN function = (CUMODULELOADDATAEX_FN)find_zluda_symbol("cuModuleLoadDataEx");
    verify_native_kernels(image);
    capture_ptx_if_present(image, "cuModuleLoadDataEx");
    CUresult result = function != NULL ? function(module, image, option_count, options, option_values)
                                       : missing("cuModuleLoadDataEx");
    if (result == CUDA_SUCCESS && option_count == 0) microcuda_k_note_module(*module, image);
    tracef("cuModuleLoadDataEx image=%p options=%u result=%d module=%p",
           image, option_count, result, module != NULL ? *module : NULL);
    return result;
}

CUresult WINAPI cuModuleLoad(CUmodule* module, const char* filename)
{
    CUMODULELOAD_FN function = (CUMODULELOAD_FN)find_zluda_symbol("cuModuleLoad");
    CUresult result = function != NULL ? function(module, filename) : missing("cuModuleLoad");
    tracef("cuModuleLoad file=%s result=%d module=%p", filename != NULL ? filename : "<null>",
           result, module != NULL ? *module : NULL);
    return result;
}

CUresult WINAPI cuModuleUnload(CUmodule module)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUMODULEUNLOAD_FN function = (CUMODULEUNLOAD_FN)find_zluda_symbol("cuModuleUnload");
    CUresult result = function != NULL ? function(module) : missing("cuModuleUnload");
    if (result == CUDA_SUCCESS) microcuda_k_remove_module(module);
    tracef("cuModuleUnload module=%p result=%d", module, result);
    return result;
}

CUresult WINAPI cuModuleGetFunction(CUfunction* function_out, CUmodule module, const char* name)
{
    CUMODULEGETFUNCTION_FN function = (CUMODULEGETFUNCTION_FN)find_zluda_symbol("cuModuleGetFunction");
    CUresult result = function != NULL ? function(function_out, module, name) : missing("cuModuleGetFunction");
    if (result == CUDA_SUCCESS && function_out != NULL)
    {
        microcuda_k_note_function(module, *function_out, name);
        remember_function_name(*function_out, name);
        note_function_lookup(*function_out, name);
    }
    tracef("cuModuleGetFunction module=%p name=%s result=%d function=%p", module,
           name != NULL ? name : "<null>", result, function_out != NULL ? *function_out : NULL);
    return result;
}

CUresult WINAPI cuLaunchKernel(CUfunction function_handle, unsigned int grid_x, unsigned int grid_y,
                               unsigned int grid_z, unsigned int block_x, unsigned int block_y,
                               unsigned int block_z, unsigned int shared_bytes, CUstream stream,
                               void** kernel_params, void** extra)
{
    static CULAUNCHKERNEL_FN function;
    if (function == NULL)
        function = (CULAUNCHKERNEL_FN)find_zluda_symbol("cuLaunchKernel");
    unsigned int sequence = 0;
    pthread_mutex_lock(&instrumentation_lock);
    sequence = ++launch_sequence;
    pthread_mutex_unlock(&instrumentation_lock);
    replay_dump_launch(sequence, function_handle, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                       shared_bytes, extra);
    note_launch(function_handle);
    PendingKernelProfile* profile = function != NULL
        ? begin_kernel_profile(sequence, function_handle, stream) : NULL;
    CUfunction native_function = microcuda_k_lookup(function_handle);
    if (native_function != NULL)
    {
        const uint64_t count = __atomic_add_fetch(&microcuda_k_launches, 1, __ATOMIC_RELAXED);
        if (count == 1 || count % 4096 == 0)
            tracef("microcuda-k actual dispatch %llu original=%p native=%p", (unsigned long long)count, function_handle, native_function);
    }
    CULAUNCHKERNEL_FN selected = native_function != NULL ? microcuda_k_launch : function;
    CUresult result = selected != NULL
        ? selected(native_function != NULL ? native_function : function_handle, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                   shared_bytes, stream, kernel_params, extra)
        : missing("cuLaunchKernel");
    if (native_function != NULL && result == CUDA_SUCCESS && microcuda_k_stats &&
        (__atomic_load_n(&microcuda_k_launches, __ATOMIC_RELAXED) % 4096) == 0)
    {
        uint64_t prep = 0, skips = 0;
        microcuda_k_stats(native_function, &prep, &skips);
        tracef("microcuda-k prep stats runs=%llu skips=%llu", (unsigned long long)prep, (unsigned long long)skips);
    }
    end_kernel_profile(profile, result, stream);
    if (result != CUDA_SUCCESS || trace_verbose())
    {
        tracef("cuLaunchKernel[%u] %s", sequence, find_function_name(function_handle));
        tracef("cuLaunchKernel function=%p grid=%u,%u,%u block=%u,%u,%u shared=%u stream=%p kernel_params=%p extra=%p result=%d",
               function_handle, grid_x, grid_y, grid_z, block_x, block_y, block_z,
               shared_bytes, stream, kernel_params, extra, result);
    }
    if (extra != NULL && trace_verbose())
    {
        for (unsigned int index = 0; index < 8; ++index)
        {
            void* item = extra[index];
            tracef("cuLaunchKernel extra[%u]=0x%llx", index, (unsigned long long)(uintptr_t)item);
            if (item == NULL)
                break;
        }
        if ((uintptr_t)extra[0] == 1 && extra[1] != NULL &&
            (uintptr_t)extra[2] == 2 && extra[3] != NULL)
        {
            const size_t argument_bytes = *(const size_t*)extra[3];
            const size_t dump_bytes = argument_bytes < 128 ? argument_bytes : 128;
            tracef("cuLaunchKernel packed_args function=%p buffer=%p bytes=%zu",
                   function_handle, extra[1], argument_bytes);
            for (size_t offset = 0; offset < dump_bytes; offset += 8)
            {
                uint64_t word = 0;
                size_t count = dump_bytes - offset < sizeof(word)
                    ? dump_bytes - offset : sizeof(word);
                memcpy(&word, (const unsigned char*)extra[1] + offset, count);
                tracef("cuLaunchKernel packed_arg[%zu]=0x%016llx",
                       offset, (unsigned long long)word);
            }
        }
    }
    if (result == CUDA_SUCCESS && launch_stats_enabled())
        summarize_launch(sequence, function_handle, extra);
    return result;
}

CUresult WINAPI cuMemAlloc(CUdeviceptr* pointer, size_t bytes)
{
    CUMEMALLOC_FN function = (CUMEMALLOC_FN)find_zluda_symbol("cuMemAlloc_v2");
    CUresult result = function != NULL ? function(pointer, bytes) : missing("cuMemAlloc_v2");
    if (result == CUDA_SUCCESS && pointer != NULL)
    {
        remember_allocation(*pointer, bytes);
        if (microcuda_k_track) microcuda_k_track(*pointer, bytes);
    }
    TRACE_CALL(result, "cuMemAlloc->v2 bytes=%zu result=%d device_ptr=0x%llx", bytes, result,
           (unsigned long long)(pointer != NULL ? *pointer : 0));
    return result;
}

CUresult WINAPI cuMemAllocHost(void** pointer, size_t bytes)
{
    /* cuMemAllocHost is cuMemHostAlloc with no flags; ZLUDA implements only
       the latter (via hipHostMalloc). */
    typedef CUresult(__attribute__((sysv_abi)) * host_alloc_type)(void**, size_t, unsigned int);
    host_alloc_type function = (host_alloc_type)find_zluda_symbol("cuMemHostAlloc");
    CUresult result = function != NULL ? function(pointer, bytes, 0) : missing("cuMemHostAlloc");
    TRACE_CALL(result, "cuMemAllocHost bytes=%zu result=%d host_ptr=%p", bytes, result,
           pointer != NULL ? *pointer : NULL);
    return result;
}

CUresult WINAPI cuMemFree(CUdeviceptr pointer)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUMEMFREE_FN function = (CUMEMFREE_FN)find_zluda_symbol("cuMemFree_v2");
    if (microcuda_k_forget) microcuda_k_forget(pointer);
    CUresult result = function != NULL ? function(pointer) : missing("cuMemFree_v2");
    if (result == CUDA_SUCCESS)
        forget_allocation(pointer);
    TRACE_CALL(result, "cuMemFree->v2 ptr=0x%llx result=%d", (unsigned long long)pointer, result);
    return result;
}

CUresult WINAPI cuMemFreeHost(void* pointer)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUMEMFREEHOST_FN function = (CUMEMFREEHOST_FN)find_zluda_symbol("cuMemFreeHost");
    CUresult result = function != NULL ? function(pointer) : missing("cuMemFreeHost");
    TRACE_CALL(result, "cuMemFreeHost ptr=%p result=%d", pointer, result);
    return result;
}

CUresult WINAPI cuMemcpy2D(const void* copy)
{
    CUMEMCPY2D_FN function = (CUMEMCPY2D_FN)find_zluda_symbol("cuMemcpy2D_v2");
    if (microcuda_k_write) microcuda_k_write(0, SIZE_MAX);
    CUresult result = function != NULL ? function(copy) : missing("cuMemcpy2D_v2");
    TRACE_CALL(result, "cuMemcpy2D->v2 descriptor=%p result=%d", copy, result);
    return result;
}

CUresult WINAPI cuMemcpyHtoDAsync(CUdeviceptr destination, const void* source, size_t bytes, CUstream stream)
{
    CUMEMCPYHTODASYNC_FN function = (CUMEMCPYHTODASYNC_FN)find_zluda_symbol("cuMemcpyHtoDAsync_v2");
    if (microcuda_k_write) microcuda_k_write(destination, bytes);
    CUresult result = function != NULL ? function(destination, source, bytes, stream) : missing("cuMemcpyHtoDAsync_v2");
    TRACE_CALL(result, "cuMemcpyHtoDAsync->v2 dst=0x%llx src=%p bytes=%zu stream=%p result=%d",
           (unsigned long long)destination, source, bytes, stream, result);
    return result;
}

CUresult WINAPI cuMemcpyDtoH(void* destination, CUdeviceptr source, size_t bytes)
{
    CUMEMCPYDTOH_FN function = (CUMEMCPYDTOH_FN)find_zluda_symbol("cuMemcpyDtoH_v2");
    CUresult result = function != NULL ? function(destination, source, bytes) : missing("cuMemcpyDtoH_v2");
    TRACE_CALL(result, "cuMemcpyDtoH->v2 dst=%p src=0x%llx bytes=%zu result=%d",
           destination, (unsigned long long)source, bytes, result);
    return result;
}

CUresult WINAPI cuArrayCreate(CUarray* array, const void* descriptor)
{
    CUARRAYCREATEV2_FN function = (CUARRAYCREATEV2_FN)find_zluda_symbol("cuArrayCreate_v2");
    if (array == NULL || descriptor == NULL)
        return 1; /* CUDA_ERROR_INVALID_VALUE */
    if (function == NULL)
        return missing("cuArrayCreate_v2");

    const CUDA_ARRAY_DESCRIPTOR_V2* source = (const CUDA_ARRAY_DESCRIPTOR_V2*)descriptor;
    CUarray created = NULL;
    CUresult result = function(&created, source);
    if (result == CUDA_SUCCESS)
    {
        *array = created;
        if (!remember_array_descriptor(created, source))
        {
            CUARRAYDESTROY_FN destroy = (CUARRAYDESTROY_FN)find_zluda_symbol("cuArrayDestroy");
            if (destroy != NULL)
                destroy(created);
            *array = NULL;
            return CUDA_ERROR_OUT_OF_MEMORY;
        }
    }
    TRACE_CALL(result, "cuArrayCreate(v2) descriptor=%p width=%zu height=%zu format=%d channels=%u result=%d array=%p",
           descriptor, source->Width, source->Height, source->Format, source->NumChannels, result, created);
    return result;
}

CUresult WINAPI cuArray3DCreate(CUarray* array, const void* descriptor)
{
    CUARRAY3DCREATEV2_FN function = (CUARRAY3DCREATEV2_FN)find_zluda_symbol("cuArray3DCreate_v2");
    if (array == NULL || descriptor == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    CUresult result = function != NULL ? function(array, descriptor) : missing("cuArray3DCreate_v2");
    int remembered = 0;
    if (result == CUDA_SUCCESS && *array != NULL)
    {
        const CUDA_ARRAY3D_DESCRIPTOR_V2* source = (const CUDA_ARRAY3D_DESCRIPTOR_V2*)descriptor;
        CUDA_ARRAY_DESCRIPTOR_V2 array_descriptor = {
            source->Width, source->Height, source->Format, source->NumChannels
        };
        remembered = remember_array_descriptor(*array, &array_descriptor);
    }
    const CUDA_ARRAY3D_DESCRIPTOR_V2* source = (const CUDA_ARRAY3D_DESCRIPTOR_V2*)descriptor;
    TRACE_CALL(result, "cuArray3DCreate->v2 width=%zu height=%zu depth=%zu format=%d channels=%u flags=0x%x descriptor=%p result=%d array=%p metadata=%s",
           source->Width, source->Height, source->Depth, source->Format,
           source->NumChannels, source->Flags, descriptor, result,
           result == CUDA_SUCCESS ? *array : NULL, remembered ? "stored" : "missing");
    return result;
}

CUresult WINAPI cuArrayDestroy(CUarray array)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUARRAYDESTROY_FN function = (CUARRAYDESTROY_FN)find_zluda_symbol("cuArrayDestroy");
    CUresult result = function != NULL ? function(array) : missing("cuArrayDestroy");
    if (result == CUDA_SUCCESS)
    {
        /* A recycled array handle must not inherit a released feature's output target. */
        pthread_mutex_lock(&instrumentation_lock);
        if (redirect_array == array)
        {
            redirect_array = NULL;
            redirect_pointer = 0;
            redirect_pitch = 0;
        }
        pthread_mutex_unlock(&instrumentation_lock);
        forget_array_descriptor(array);
    }
    TRACE_CALL(result, "cuArrayDestroy array=%p result=%d", array, result);
    return result;
}

/* d4r linear inputs: pitch-linear texture objects the shim builds on its interop buffers. NGX validates
   input textures through cuTexObjectGetResourceDesc + cuArrayGetDescriptor, so these are reported as
   arrays (a stand-in handle per texture) with the plane's size and format; the texture itself is used
   unchanged in NGX's kernels. */
typedef struct LinearTexture
{
    CUtexObject object;
    void* fake_array;
    size_t width, height;
    uint32_t format, channels;
    struct LinearTexture* next;
} LinearTexture;
static LinearTexture* linear_textures;
static pthread_mutex_t linear_lock = PTHREAD_MUTEX_INITIALIZER;

CUresult WINAPI d4rRegisterLinearTexture(CUtexObject object, size_t width, size_t height, uint32_t format,
                                         uint32_t channels)
{
    LinearTexture* entry = (LinearTexture*)calloc(1, sizeof(*entry));
    if (entry == NULL)
        return CUDA_ERROR_OUT_OF_MEMORY;
    entry->object = object;
    entry->fake_array = malloc(16);
    entry->width = width;
    entry->height = height;
    entry->format = format;
    entry->channels = channels;
    pthread_mutex_lock(&linear_lock);
    entry->next = linear_textures;
    linear_textures = entry;
    pthread_mutex_unlock(&linear_lock);
    tracef("d4rRegisterLinearTexture object=0x%llx %zux%zu format=%u channels=%u stand-in array=%p",
           (unsigned long long)object, width, height, format, channels, entry->fake_array);
    return CUDA_SUCCESS;
}

static int linear_texture_lookup(CUtexObject object, void* fake_array, LinearTexture* out)
{
    int found = 0;
    pthread_mutex_lock(&linear_lock);
    for (LinearTexture* entry = linear_textures; entry != NULL && !found; entry = entry->next)
        if ((object != 0 && entry->object == object) || (fake_array != NULL && entry->fake_array == fake_array))
        {
            *out = *entry;
            found = 1;
        }
    pthread_mutex_unlock(&linear_lock);
    return found;
}

static void linear_texture_forget(CUtexObject object)
{
    pthread_mutex_lock(&linear_lock);
    for (LinearTexture** link = &linear_textures; *link != NULL; link = &(*link)->next)
        if ((*link)->object == object)
        {
            LinearTexture* gone = *link;
            *link = gone->next;
            free(gone->fake_array);
            free(gone);
            break;
        }
    pthread_mutex_unlock(&linear_lock);
}

CUresult WINAPI cuArrayGetDescriptor(void* descriptor, CUarray array)
{
    LinearTexture linear;
    if (descriptor != NULL && linear_texture_lookup(0, (void*)array, &linear))
    {
        CUDA_ARRAY_DESCRIPTOR_V2* output = (CUDA_ARRAY_DESCRIPTOR_V2*)descriptor;
        output->Width = linear.width;
        output->Height = linear.height;
        output->Format = linear.format;
        output->NumChannels = linear.channels;
        TRACE_CALL(0, "cuArrayGetDescriptor array=%p (linear texture stand-in) width=%zu height=%zu", array,
                   linear.width, linear.height);
        return CUDA_SUCCESS;
    }
    if (descriptor == NULL || array == NULL)
        return CUDA_ERROR_INVALID_VALUE;

    CUDA_ARRAY_DESCRIPTOR_V2* output = (CUDA_ARRAY_DESCRIPTOR_V2*)descriptor;
    if (lookup_array_descriptor(array, output))
    {
        TRACE_CALL(0, "cuArrayGetDescriptor array=%p width=%zu height=%zu format=%d channels=%u result=0 (bridge metadata)",
               array, output->Width, output->Height, output->Format, output->NumChannels);
        return CUDA_SUCCESS;
    }

    CUARRAYGETDESCRIPTORV2_FN function =
        (CUARRAYGETDESCRIPTORV2_FN)find_zluda_symbol("cuArrayGetDescriptor_v2");
    CUresult result = function != NULL ? function(output, array) : missing("cuArrayGetDescriptor_v2");
    tracef("cuArrayGetDescriptor array=%p result=%d", array, result);
    return result;
}

CUresult WINAPI cuMipmappedArrayDestroy(CUmipmappedArray array)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUMIPMAPPEDARRAYDESTROY_FN function =
        (CUMIPMAPPEDARRAYDESTROY_FN)find_zluda_symbol("cuMipmappedArrayDestroy");
    CUresult result = function != NULL ? function(array) : missing("cuMipmappedArrayDestroy");
    tracef("cuMipmappedArrayDestroy array=%p result=%d", array, result);
    return result;
}

CUresult WINAPI cuDestroyExternalMemory(CUexternalMemory memory)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUEXTERNALMEMORYDESTROY_FN function =
        (CUEXTERNALMEMORYDESTROY_FN)find_zluda_symbol("cuDestroyExternalMemory");
    CUresult result = function != NULL ? function(memory) : missing("cuDestroyExternalMemory");
    tracef("cuDestroyExternalMemory memory=%p result=%d", memory, result);
    return result;
}

CUresult WINAPI cuSurfObjectDestroy(CUsurfObject object);

CUresult WINAPI cuSurfObjectCreate(CUsurfObject* object, const void* descriptor)
{
    CUSURFOBJECTCREATE_FN function = (CUSURFOBJECTCREATE_FN)find_zluda_symbol("cuSurfObjectCreate");
    CUresult result = function != NULL ? function(object, descriptor) : missing("cuSurfObjectCreate");
    if (result == CUDA_SUCCESS && object != NULL && descriptor != NULL)
    {
        remember_surface_object(*object, descriptor);
        const CUarray array = find_surface_array(*object);
        pthread_mutex_lock(&instrumentation_lock);
        const int redirected = redirect_array != NULL && array == redirect_array;
        const CUdeviceptr pointer = redirect_pointer;
        const uint32_t pitch = redirect_pitch;
        pthread_mutex_unlock(&instrumentation_lock);
        if (redirected && (result = write_redirect_tail(*object, pointer, pitch, 0)) != CUDA_SUCCESS)
            cuSurfObjectDestroy(*object);
    }
    TRACE_CALL(result, "cuSurfObjectCreate descriptor=%p result=%d object=0x%llx", descriptor, result,
           (unsigned long long)(object != NULL ? *object : 0));
    return result;
}

CUresult WINAPI cuSurfObjectDestroy(CUsurfObject object)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUSURFOBJECTDESTROY_FN function = (CUSURFOBJECTDESTROY_FN)find_zluda_symbol("cuSurfObjectDestroy");
    pthread_mutex_lock(&instrumentation_lock);
    for (SurfaceObjectRecord** link = &surface_objects; *link != NULL; link = &(*link)->next)
    {
        if ((*link)->object == object)
        {
            SurfaceObjectRecord* gone = *link;
            *link = gone->next;
            free(gone);
            break;
        }
    }
    pthread_mutex_unlock(&instrumentation_lock);
    CUresult result = function != NULL ? function(object) : missing("cuSurfObjectDestroy");
    TRACE_CALL(result, "cuSurfObjectDestroy object=0x%llx result=%d", (unsigned long long)object, result);
    return result;
}

CUresult WINAPI cuSurfObjectGetResourceDesc(void* descriptor, CUsurfObject object)
{
    CUSURFOBJECTGETDESC_FN function =
        (CUSURFOBJECTGETDESC_FN)find_zluda_symbol("cuSurfObjectGetResourceDesc");
    CUresult result = function != NULL ? function(descriptor, object) : missing("cuSurfObjectGetResourceDesc");
    TRACE_CALL(result, "cuSurfObjectGetResourceDesc object=0x%llx result=%d", (unsigned long long)object, result);
    return result;
}

CUresult WINAPI cuTexObjectCreate(CUtexObject* object, const void* resource,
                                  const void* texture, const void* view)
{
    CUTEXOBJECTCREATE_FN function = (CUTEXOBJECTCREATE_FN)find_zluda_symbol("cuTexObjectCreate");
    CUresult result = function != NULL ? function(object, resource, texture, view) : missing("cuTexObjectCreate");
    uint32_t resource_type = UINT32_MAX;
    uint64_t array_handle = 0;
    uint32_t address0 = UINT32_MAX;
    uint32_t address1 = UINT32_MAX;
    uint32_t filter_mode = UINT32_MAX;
    uint32_t texture_flags = UINT32_MAX;
    uint32_t max_anisotropy = UINT32_MAX;
    uint32_t mipmap_filter = UINT32_MAX;
    if (resource != NULL)
    {
        memcpy(&resource_type, resource, sizeof(resource_type));
        memcpy(&array_handle, (const unsigned char*)resource + 8, sizeof(array_handle));
    }
    if (texture != NULL)
    {
        memcpy(&address0, texture, sizeof(address0));
        memcpy(&address1, (const unsigned char*)texture + 4, sizeof(address1));
        memcpy(&filter_mode, (const unsigned char*)texture + 12, sizeof(filter_mode));
        memcpy(&texture_flags, (const unsigned char*)texture + 16, sizeof(texture_flags));
        memcpy(&max_anisotropy, (const unsigned char*)texture + 20, sizeof(max_anisotropy));
        memcpy(&mipmap_filter, (const unsigned char*)texture + 24, sizeof(mipmap_filter));
    }
    if (result == CUDA_SUCCESS && object != NULL && *object != 0 && resource != NULL &&
        !remember_texture_descriptor(*object, resource))
        tracef("cuTexObjectCreate descriptor tracking failed object=0x%llx",
               (unsigned long long)*object);
    TRACE_CALL(result, "cuTexObjectCreate resource=%p type=%u array=0x%llx texture=%p addr=%u,%u filter=%u flags=0x%x aniso=%u mipfilter=%u view=%p result=%d object=0x%llx",
           resource, resource_type, (unsigned long long)array_handle, texture, address0, address1,
           filter_mode, texture_flags, max_anisotropy, mipmap_filter, view, result,
           (unsigned long long)(object != NULL ? *object : 0));
    return result;
}

CUresult WINAPI cuTexObjectDestroy(CUtexObject object)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    CUTEXOBJECTDESTROY_FN function = (CUTEXOBJECTDESTROY_FN)find_zluda_symbol("cuTexObjectDestroy");
    CUresult result = function != NULL ? function(object) : missing("cuTexObjectDestroy");
    if (result == CUDA_SUCCESS)
    {
        forget_texture_descriptor(object);
        linear_texture_forget(object);
    }
    TRACE_CALL(result, "cuTexObjectDestroy object=0x%llx result=%d", (unsigned long long)object, result);
    return result;
}

CUresult WINAPI cuTexObjectGetResourceDesc(void* descriptor, CUtexObject object)
{
    LinearTexture linear;
    if (descriptor != NULL && linear_texture_lookup(object, NULL, &linear))
    {
        memset(descriptor, 0, 144); // CUDA_RESOURCE_DESC: resType ARRAY, res.array.hArray at 8
        memcpy((unsigned char*)descriptor + 8, &linear.fake_array, sizeof(void*));
        TRACE_CALL(0, "cuTexObjectGetResourceDesc object=0x%llx -> linear texture stand-in array %p",
                   (unsigned long long)object, linear.fake_array);
        return CUDA_SUCCESS;
    }
    CUTEXOBJECTGETDESC_FN function =
        (CUTEXOBJECTGETDESC_FN)find_zluda_symbol("cuTexObjectGetResourceDesc");
    CUresult result = function != NULL ? function(descriptor, object) : CUDA_ERROR_NOT_SUPPORTED;
    if (result != CUDA_SUCCESS && lookup_texture_descriptor(object, descriptor))
        result = CUDA_SUCCESS;
    TRACE_CALL(0, "cuTexObjectGetResourceDesc descriptor=%p object=0x%llx result=%d",
           descriptor, (unsigned long long)object, result);
    return result;
}


/* Direct pass-throughs for driver APIs whose arguments are all integers or
   pointers, so the Windows x64 call forwards unchanged to ZLUDA (SysV). Unversioned
   names resolve to the versions the CUDA 12 driver returns for them. */
static void microcuda_k_notify_generic_write(const char* target)
{
    if (microcuda_k_write && (strncmp(target, "cuMemcpyHtoD", 12) == 0 || strncmp(target, "cuMemcpyDtoD", 12) == 0 || strncmp(target, "cuMemset", 8) == 0))
        microcuda_k_write(0, SIZE_MAX); /* unsupported/general writer: invalidate all prepared weights */
}
typedef uintptr_t D4rArg;
static int d4r_forward_is_teardown(const char* target)
{
    return strstr(target, "Destroy") != NULL || strstr(target, "Synchronize") != NULL || strstr(target, "Release") != NULL;
}
#define D4R_FORWARD(name, target, count, params, args) \
    CUresult WINAPI name params \
    { \
        if (d4r_forward_is_teardown(#target) && before_teardown()) \
            return CUDA_SUCCESS; \
        typedef CUresult(__attribute__((sysv_abi)) * function_type) params; \
        function_type function = (function_type)find_zluda_symbol(#target); \
        microcuda_k_notify_generic_write(#target); \
        CUresult result = function != NULL ? function args : missing(#target); \
        TRACE_CALL(result, #name "->" #target " result=%d", result); \
        return result; \
    }
D4R_FORWARD(cuEventCreate, cuEventCreate, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuEventDestroy, cuEventDestroy_v2, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuEventDestroy_v2, cuEventDestroy_v2, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuEventRecord, cuEventRecord, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuEventRecordWithFlags, cuEventRecordWithFlags, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
CUresult WINAPI cuEventSynchronize(D4rArg a0)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    typedef CUresult(__attribute__((sysv_abi)) * function_type)(D4rArg);
    if (elide_ngx_sync())
    {
        TRACE_CALL(CUDA_SUCCESS, "cuEventSynchronize elided");
        return CUDA_SUCCESS;
    }
    function_type function = (function_type)find_zluda_symbol("cuEventSynchronize");
    CUresult result = function != NULL ? function(a0) : missing("cuEventSynchronize");
    TRACE_CALL(result, "cuEventSynchronize->cuEventSynchronize result=%d", result);
    return result;
}
/* The shim's output wait must not inherit D4R_ELIDE_NGX_SYNC: it releases the
   game's split command list only after this event has actually completed. */
CUresult WINAPI d4rEventSynchronize(D4rArg a0)
{
    if (before_teardown())
        return CUDA_SUCCESS;
    typedef CUresult(__attribute__((sysv_abi)) * function_type)(D4rArg);
    function_type function = (function_type)find_zluda_symbol("cuEventSynchronize");
    CUresult result = function != NULL ? function(a0) : missing("cuEventSynchronize");
    if (result == CUDA_SUCCESS)
        flush_kernel_profiles();
    TRACE_CALL(result, "d4rEventSynchronize result=%d", result);
    return result;
}
CUresult WINAPI cuEventQuery(D4rArg event)
{
    typedef CUresult(__attribute__((sysv_abi)) *function_type)(D4rArg);
    function_type function = (function_type)find_zluda_symbol("cuEventQuery");
    CUresult result = function != NULL ? function(event) : missing("cuEventQuery");
    /* NOT_READY is the normal result while the shim sleeps between queries,
       not an error to flush to the log on every poll. */
    TRACE_CALL(result == CUDA_ERROR_NOT_READY ? CUDA_SUCCESS : result,
               "cuEventQuery result=%d", result);
    return result;
}
/* With elided event waits an event may still be pending: report 0 ms instead of CUDA_ERROR_NOT_READY. */
CUresult WINAPI cuEventElapsedTime(D4rArg a0, D4rArg a1, D4rArg a2)
{
    typedef CUresult(__attribute__((sysv_abi)) * function_type)(D4rArg, D4rArg, D4rArg);
    function_type function = (function_type)find_zluda_symbol("cuEventElapsedTime");
    CUresult result = function != NULL ? function(a0, a1, a2) : missing("cuEventElapsedTime");
    if (result == CUDA_ERROR_NOT_READY && elide_ngx_sync() && a0 != 0)
    {
        *(float*)(uintptr_t)a0 = 0.0f;
        result = CUDA_SUCCESS;
    }
    TRACE_CALL(result, "cuEventElapsedTime->cuEventElapsedTime result=%d", result);
    return result;
}
D4R_FORWARD(cuStreamCreate, cuStreamCreate, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuStreamCreateWithPriority, cuStreamCreateWithPriority, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuStreamDestroy, cuStreamDestroy_v2, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuStreamDestroy_v2, cuStreamDestroy_v2, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuStreamSynchronize, cuStreamSynchronize, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuStreamQuery, cuStreamQuery, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuStreamWaitEvent, cuStreamWaitEvent, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuStreamGetFlags, cuStreamGetFlags, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuStreamGetPriority, cuStreamGetPriority, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuCtxGetCurrent, cuCtxGetCurrent, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuCtxGetStreamPriorityRange, cuCtxGetStreamPriorityRange, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuCtxGetLimit, cuCtxGetLimit, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuCtxSetLimit, cuCtxSetLimit, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuDriverGetVersion, cuDriverGetVersion, 1, (D4rArg a0), (a0))
D4R_FORWARD(cuDeviceGetName, cuDeviceGetName, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuDeviceTotalMem, cuDeviceTotalMem_v2, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuDeviceTotalMem_v2, cuDeviceTotalMem_v2, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuDeviceComputeCapability, cuDeviceComputeCapability, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemGetInfo, cuMemGetInfo_v2, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuMemGetInfo_v2, cuMemGetInfo_v2, 2, (D4rArg a0, D4rArg a1), (a0, a1))
D4R_FORWARD(cuMemcpyHtoD, cuMemcpyHtoD_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemcpyHtoD_v2, cuMemcpyHtoD_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemcpyDtoD, cuMemcpyDtoD_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemcpyDtoDAsync, cuMemcpyDtoDAsync_v2, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuMemcpyDtoHAsync, cuMemcpyDtoHAsync_v2, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuMemsetD8, cuMemsetD8_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemsetD16, cuMemsetD16_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemsetD32, cuMemsetD32_v2, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuMemsetD8Async, cuMemsetD8Async, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuMemsetD16Async, cuMemsetD16Async, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuMemsetD32Async, cuMemsetD32Async, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuMemsetD2D8, cuMemsetD2D8_v2, 5, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3, D4rArg a4), (a0, a1, a2, a3, a4))
D4R_FORWARD(cuMemsetD2D32, cuMemsetD2D32_v2, 5, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3, D4rArg a4), (a0, a1, a2, a3, a4))
D4R_FORWARD(cuFuncSetAttribute, cuFuncSetAttribute, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuFuncGetAttribute, cuFuncGetAttribute, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuModuleGetGlobal, cuModuleGetGlobal_v2, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuOccupancyMaxActiveBlocksPerMultiprocessor, cuOccupancyMaxActiveBlocksPerMultiprocessor, 4, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3), (a0, a1, a2, a3))
D4R_FORWARD(cuOccupancyMaxPotentialBlockSize, cuOccupancyMaxPotentialBlockSize, 6, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3, D4rArg a4, D4rArg a5), (a0, a1, a2, a3, a4, a5))
D4R_FORWARD(cuMemAllocPitch, cuMemAllocPitch_v2, 5, (D4rArg a0, D4rArg a1, D4rArg a2, D4rArg a3, D4rArg a4), (a0, a1, a2, a3, a4))
D4R_FORWARD(cuMemHostAlloc, cuMemHostAlloc, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))
D4R_FORWARD(cuPointerGetAttribute, cuPointerGetAttribute, 3, (D4rArg a0, D4rArg a1, D4rArg a2), (a0, a1, a2))

CUresult WINAPI cuGetProcAddress_v2(const char* symbol, void** pointer, int cuda_version,
                                    uint64_t flags, int* symbol_status)
{
    void* bridge = symbol != NULL ? bridge_export(symbol) : NULL;
    if (pointer != NULL)
        *pointer = bridge;
    if (symbol_status != NULL)
        *symbol_status = bridge != NULL ? 0 : 1;
    tracef("cuGetProcAddress_v2 name=%s cudaVersion=%d flags=0x%llx bridge=%p",
           symbol != NULL ? symbol : "<null>", cuda_version, (unsigned long long)flags, bridge);
    return bridge != NULL ? CUDA_SUCCESS : CUDA_ERROR_NOT_FOUND;
}

CUresult WINAPI cuGetProcAddress(const char* symbol, void** pointer, int cuda_version, uint64_t flags)
{
    return cuGetProcAddress_v2(symbol, pointer, cuda_version, flags, NULL);
}

/* Experimental zero-copy interop (tools/d3d12_hip_interop_probe.cpp).
   Wine's D3D12 shared NT handles carry no unix fd (the wineserver shared
   resource object has no get_fd) and vkd3d-proton only exports textures, so
   interop goes through Vulkan instead: the caller allocates a VkBuffer's
   memory on vkd3d-proton's VkDevice with VkExportMemoryAllocateInfo, which
   winevulkan turns into a host OPAQUE_FD export. Here, on the unix side of the
   same process, the winevulkan client handles are translated into host
   handles (Wine 11 layout: a VkDevice client object holds the unix object
   pointer at offset 8; a non-dispatchable client handle is the unix object
   pointer; every unix object starts with its host handle), the host fd is
   exported with vkGetMemoryFdKHR and imported into HIP. HIP does not take
   ownership of the fd (hipDestroyExternalMemory leaves it open, and an open fd
   keeps the memory alive after vkFreeMemory), so it is closed once imported, as
   ROCm's own GL interop does after mapping. */
typedef struct
{
    int type; /* hipExternalMemoryHandleTypeOpaqueFd = 1 */
    union
    {
        int fd;
        struct
        {
            void* handle;
            const void* name;
        } win32;
        const void* nvSciBufObject;
    } handle;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} D4rHipExternalMemoryHandleDesc;

typedef struct
{
    unsigned long long offset;
    unsigned long long size;
    unsigned int flags;
    unsigned int reserved[16];
} D4rHipExternalMemoryBufferDesc;

typedef struct
{
    int sType; /* VK_STRUCTURE_TYPE_MEMORY_GET_FD_INFO_KHR = 1000074002 */
    const void* pNext;
    uint64_t memory;
    int handleType; /* VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_FD_BIT = 1 */
} D4rVkMemoryGetFdInfo;

typedef int(__attribute__((sysv_abi)) * HIP_IMPORT_EXTERNAL_MEMORY_FN)(void**, const D4rHipExternalMemoryHandleDesc*);
typedef int(__attribute__((sysv_abi)) * HIP_EXTERNAL_MEMORY_GET_MAPPED_BUFFER_FN)(void**, void*,
                                                                                const D4rHipExternalMemoryBufferDesc*);
typedef int(__attribute__((sysv_abi)) * HIP_DESTROY_EXTERNAL_MEMORY_FN)(void*);
typedef void*(__attribute__((sysv_abi)) * VK_GET_DEVICE_PROC_ADDR_FN)(void*, const char*);
typedef int(__attribute__((sysv_abi)) * VK_GET_MEMORY_FD_FN)(void*, const D4rVkMemoryGetFdInfo*, int*);

static void* hip_symbol(const char* name)
{
    static void* hip;
    if (hip == NULL)
        hip = dlopen("libamdhip64.so.7", RTLD_NOW | RTLD_NOLOAD);
    if (hip == NULL)
        hip = dlopen("libamdhip64.so", RTLD_NOW | RTLD_NOLOAD);
    return hip != NULL ? dlsym(hip, name) : NULL;
}

CUresult WINAPI d4rImportVulkanMemory(void* client_device, uint64_t client_memory, uint64_t bytes, CUdeviceptr* pointer,
                                      void** memory)
{
    ensure_context();
    if (client_device == NULL || client_memory == 0 || pointer == NULL || memory == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    HIP_IMPORT_EXTERNAL_MEMORY_FN import = (HIP_IMPORT_EXTERNAL_MEMORY_FN)hip_symbol("hipImportExternalMemory");
    HIP_EXTERNAL_MEMORY_GET_MAPPED_BUFFER_FN map =
        (HIP_EXTERNAL_MEMORY_GET_MAPPED_BUFFER_FN)hip_symbol("hipExternalMemoryGetMappedBuffer");
    HIP_DESTROY_EXTERNAL_MEMORY_FN destroy = (HIP_DESTROY_EXTERNAL_MEMORY_FN)hip_symbol("hipDestroyExternalMemory");
    void* vulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    VK_GET_DEVICE_PROC_ADDR_FN get_proc =
        vulkan != NULL ? (VK_GET_DEVICE_PROC_ADDR_FN)dlsym(vulkan, "vkGetDeviceProcAddr") : NULL;
    if (import == NULL || map == NULL || destroy == NULL || get_proc == NULL)
    {
        tracef("d4rImportVulkanMemory: HIP external memory or host Vulkan unavailable");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    /* Each unix object is {host handle, client handle, ...}; checking the
       client handle guards against a different winevulkan layout. */
    const uint64_t* device_object = (const uint64_t*)(uintptr_t)((const uint64_t*)client_device)[1];
    const uint64_t* memory_object = (const uint64_t*)(uintptr_t)client_memory;
    if (device_object == NULL || device_object[1] != (uint64_t)(uintptr_t)client_device ||
        memory_object[1] != client_memory)
    {
        tracef("d4rImportVulkanMemory: unexpected winevulkan object layout");
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    void* host_device = (void*)(uintptr_t)device_object[0];
    const uint64_t host_memory = memory_object[0];
    VK_GET_MEMORY_FD_FN get_fd = (VK_GET_MEMORY_FD_FN)get_proc(host_device, "vkGetMemoryFdKHR");
    if (get_fd == NULL)
    {
        tracef("d4rImportVulkanMemory: vkGetMemoryFdKHR unavailable on host device %p", host_device);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    D4rVkMemoryGetFdInfo info = {1000074002, NULL, host_memory, 1};
    int fd = -1;
    int vk = get_fd(host_device, &info, &fd);
    if (vk != 0 || fd < 0)
    {
        tracef("d4rImportVulkanMemory: vkGetMemoryFdKHR(device %p, memory 0x%llx) failed: %d", host_device,
               (unsigned long long)host_memory, vk);
        return CUDA_ERROR_INVALID_VALUE;
    }
    D4rHipExternalMemoryHandleDesc desc;
    memset(&desc, 0, sizeof(desc));
    desc.type = 1;
    desc.handle.fd = fd;
    desc.size = bytes;
    desc.flags = 1; /* dedicated allocation */
    void* external = NULL;
    int result = import(&external, &desc);
    close(fd);
    if (result != 0)
    {
        tracef("d4rImportVulkanMemory: hipImportExternalMemory(fd=%d, %llu bytes) failed: %d", fd,
               (unsigned long long)bytes, result);
        return CUDA_ERROR_INVALID_VALUE;
    }
    D4rHipExternalMemoryBufferDesc buffer;
    memset(&buffer, 0, sizeof(buffer));
    buffer.size = bytes;
    void* device = NULL;
    result = map(&device, external, &buffer);
    if (result != 0)
    {
        tracef("d4rImportVulkanMemory: hipExternalMemoryGetMappedBuffer failed: %d", result);
        destroy(external);
        return CUDA_ERROR_INVALID_VALUE;
    }
    *pointer = (CUdeviceptr)(uintptr_t)device;
    *memory = external;
    tracef("d4rImportVulkanMemory: host memory 0x%llx -> fd %d -> device %p (%llu bytes)",
           (unsigned long long)host_memory, fd, device, (unsigned long long)bytes);
    return CUDA_SUCCESS;
}

/* d4r engine: the Swin network on d4r's native HIP layer kernels, launched directly through HIP (no NGX, no
   PTX; engine/hip_net.cpp is the same logic for native Linux programs). The engine's Vulkan front writes the
   tokens into `tokens`, the network's layers run here, and its back reads `output`; both are device pointers of
   memory imported with d4rImportVulkanMemory. `net` is the model's hipnet.bin:
     D4RMHIP1 (compile_m.py --hip): preset M, ten layers, width/height are the render dimensions; two immutable
       window-alignment graphs, one per evaluation.
     D4RKHIP1 (compile_k.py --hip): preset K, eleven layers (the bottleneck dec5 as its main kernel plus
       post1..3), width/height are the output dimensions (the token grid is makeKShape), and `tokens`/`output`
       hold 16 / 40 f16 channels per token row-major. Eight immutable graphs, one per makeKWindows evaluation.
   Either way: {kernel name[32], shift x, y, grid divisor, pad, weight offset, weight bytes} per layer, then the
   weight allocations as NGX uploads them. */
typedef struct
{
    const char* name; /* kernel name, as in hipnet.bin */
    const void* data; /* its code object (.hsaco) */
    uint64_t bytes;
} D4rEngineNetCode;

typedef struct
{
    void* w;
    int32_t sx, sy, tw, th;
    void* in;
    void* skip;
    void* out;
    void* merged;
    int32_t r69, r70, r71, r72;
    void* in_flags;
    void* out_flags;
} D4rEngineNetParams; /* kernels/m/swin_common.h: CommonParams, TubeParams */

typedef struct
{
    int32_t W, H; /* token grid of the layer, pixels */
    uint8_t* in;
    uint8_t* skip;  /* decoders: the full-resolution encoder output */
    uint8_t* out24; /* encoders: patch-merged output; decoders: output; dec0: the caller's head buffer */
    uint8_t* out32; /* encoders: full-resolution output (the skip of the matching decoder) */
    uint64_t pad40, pad48;
    int32_t sx, sy;
    uint8_t* w;
    uint8_t rest[104];
} D4rEngineNetPwinParams; /* kernels/k/pwin_layer.h: PwinParams */
_Static_assert(sizeof(D4rEngineNetPwinParams) == 176, "K param block");

/* one immutable launch of one graph: entry point, grid, block and the frozen parameter block (the active
   member follows the model's magic) */
typedef union
{
    D4rEngineNetParams m;
    D4rEngineNetPwinParams k;
} D4rEngineNetParamsU;

typedef struct
{
    void* function;
    uint32_t gx, gy, bx, by, bz;
    D4rEngineNetParamsU params;
} D4rEngineNetNode;

#define D4R_ENGINE_NET_M_LAYERS 10
#define D4R_ENGINE_NET_K_LAYERS 11
#define D4R_ENGINE_NET_LAYERS D4R_ENGINE_NET_K_LAYERS
#define D4R_ENGINE_NET_MODULES D4R_ENGINE_NET_K_LAYERS
#define D4R_ENGINE_NET_GRAPHS 8
#define D4R_ENGINE_NET_NODES 16 /* 11 layers, the deepest of them with its three post phases */
#define D4R_ENGINE_NET_OWNED 32
#define D4R_ENGINE_NET_SIGNALS 16
struct D4rEngineNet;
typedef struct
{
    struct D4rEngineNet* net;
    uint64_t value;
    int busy;
} D4rEngineNetSignal;

typedef struct
{
    int sType; /* VK_STRUCTURE_TYPE_SEMAPHORE_SIGNAL_INFO = 1000207005 */
    const void* pNext;
    uint64_t semaphore, value;
} D4rVkSemaphoreSignalInfo;
typedef int (*VK_SIGNAL_SEMAPHORE_FN)(void*, const D4rVkSemaphoreSignalInfo*);
typedef int (*VK_GET_SEMAPHORE_COUNTER_FN)(void*, uint64_t, uint64_t*);
typedef int (*HIP_LAUNCH_HOST_FN)(void*, void (*)(void*), void*);

typedef struct D4rEngineNet
{
    int k_preset;  /* D4RKHIP1: the token grid comes from the output dimensions and the cycle has eight graphs */
    uint32_t graphs; /* 2 for M, 8 for K */
    void* modules[D4R_ENGINE_NET_MODULES];
    void* owned[D4R_ENGINE_NET_OWNED];
    int module_count, owned_count;
    uint32_t node_count[D4R_ENGINE_NET_GRAPHS];
    D4rEngineNetNode nodes[D4R_ENGINE_NET_GRAPHS][D4R_ENGINE_NET_NODES];
    /* weight-image preparations: one per layer, i.e. per distinct weights pointer, run once at create time */
    uint32_t prep_count;
    D4rEngineNetNode preps[D4R_ENGINE_NET_LAYERS];
    void* stream;
    void* graph[D4R_ENGINE_NET_GRAPHS];
    void* executable[D4R_ENGINE_NET_GRAPHS];
    uint32_t frame;
    uint32_t* input_wait_host;
    void* start;
    void* stop;
    /* Input readiness on the GPU (d4rEngineNetInputWaitReady / d4rEngineNetLaunchAt): the u32 the engine's
       front fills with the frame number, and the value a queued stream wait is waiting for (0: none). */
    CUdeviceptr input_wait;
    uint32_t input_wait_value;
    void* timeline_device;
    uint64_t timeline_semaphore, timeline_value;
    pthread_mutex_t timeline_lock;
    VK_SIGNAL_SEMAPHORE_FN signal_timeline;
    HIP_LAUNCH_HOST_FN launch_host;
    D4rEngineNetSignal signals[D4R_ENGINE_NET_SIGNALS];
    int timeline_error;
} D4rEngineNet;

typedef int (*HIP_PP_FN)(void**);
typedef int (*HIP_P_FN)(void*);
typedef int (*HIP_MALLOC_FN)(void**, size_t);
typedef int (*HIP_MEMCPY_FN)(void*, const void*, size_t);
typedef int (*HIP_MODULE_LOAD_DATA_FN)(void**, const void*);
typedef int (*HIP_MODULE_GET_FUNCTION_FN)(void**, void*, const char*);
typedef int (*HIP_MODULE_GET_GLOBAL_FN)(void**, size_t*, void*, const char*);
typedef int (*HIP_MODULE_LAUNCH_FN)(void*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, void*,
                                    void**, void**);
typedef int (*HIP_EVENT_RECORD_FN)(void*, void*);
typedef int (*HIP_EVENT_ELAPSED_FN)(float*, void*, void*);
typedef int (*HIP_STREAM_WAIT_VALUE32_FN)(void*, void*, uint32_t, unsigned int, uint32_t);
typedef int (*HIP_STREAM_WRITE_VALUE32_FN)(void*, void*, uint32_t, unsigned int);
typedef int (*HIP_STREAM_CREATE_WITH_FLAGS_FN)(void**, unsigned int);
typedef int (*HIP_STREAM_SYNCHRONIZE_FN)(void*);
typedef int (*HIP_CAPTURE_BEGIN_FN)(void*, int);
typedef int (*HIP_CAPTURE_END_FN)(void*, void**);
typedef int (*HIP_GRAPH_INSTANTIATE_FN)(void**, void*, void**, char*, size_t);


CUresult WINAPI d4rEngineNetReleaseInput(void* handle, uint32_t value)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    if (net == NULL || net->input_wait_host == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    /* Never rewind a newer front's publication when releasing a discarded older frame. */
    uint32_t current = __atomic_load_n(net->input_wait_host, __ATOMIC_ACQUIRE);
    while (current < value &&
           !__atomic_compare_exchange_n(net->input_wait_host, &current, value, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED))
    {}
    return CUDA_SUCCESS;
}

/* preset K's eleven layers in network order (the same plan as engine/hip_net.cpp): token-grid divisor, input
   channels, the core channels (the encoder's full-resolution output width), the channels written to out24 (the
   patch-merged skip for encoders, the block output for the decoders, the 40-channel head for dec0) and the
   window-schedule role. */
typedef struct
{
    const char* name;
    uint32_t divisor, cin, c, out24ch;
    int role;
} D4rEngineNetKLayer;

static const D4rEngineNetKLayer d4r_engine_net_k_plan[D4R_ENGINE_NET_K_LAYERS] = {
    {"enc0", 1, 16, 32, 64, -1}, {"enc1", 2, 64, 64, 64, -1},  {"enc2", 4, 64, 64, 96, -1},
    {"enc3", 8, 96, 96, 128, 2}, {"enc4", 16, 128, 128, 160, 1}, {"dec5", 32, 160, 160, 160, 0},
    {"dec4", 16, 160, 128, 128, 1}, {"dec3", 8, 128, 96, 96, 2}, {"dec2", 4, 96, 64, 64, -1},
    {"dec1", 2, 64, 64, 64, -1}, {"dec0", 1, 64, 32, 40, -1},
};

/* makeKWindows (engine/runtime.cpp): dec5 moves once per frame around an eight-frame cycle and every shallower
   layer follows the one below it as (2 * shift + 4) mod 8; role 0 = dec5, 1 = enc4/dec4, 2 = enc3/dec3,
   everything else keeps shift 4. */
static const int32_t d4r_engine_net_k_cycle[8][2] = {{0, 2}, {5, 6}, {4, 0}, {1, 4}, {7, 5}, {2, 1}, {3, 7}, {6, 3}};

static void d4r_engine_net_k_shift(int role, uint32_t frame, int32_t* sx, int32_t* sy)
{
    if (role < 0)
    {
        *sx = 4;
        *sy = 4;
        return;
    }
    int32_t x = d4r_engine_net_k_cycle[frame % 8][0], y = d4r_engine_net_k_cycle[frame % 8][1];
    for (int depth = 0; depth < role; ++depth)
    {
        x = (2 * x + 4) & 7;
        y = (2 * y + 4) & 7;
    }
    *sx = x;
    *sy = y;
}

/* one code object of a layer: the main entry, its weight-image preparation and the deep layers' post phases */
typedef struct
{
    char name[33];
    void* module;
    void* main;
    void* prep;
    void* post[8];
    uint32_t prep_blocks, block_z, grid_x, grid_y;
    uint32_t post_gx[8], post_gy[8], post_bx[8], post_by[8], post_bz[8];
    int post_count;
} D4rEngineNetKernel;

static uint32_t d4r_engine_net_global(void* module, const char* name, uint32_t fallback)
{
    HIP_MODULE_GET_GLOBAL_FN get = (HIP_MODULE_GET_GLOBAL_FN)hip_symbol("hipModuleGetGlobal");
    HIP_MEMCPY_FN to_host = (HIP_MEMCPY_FN)hip_symbol("hipMemcpyDtoH");
    void* at = NULL;
    size_t bytes = 0;
    uint32_t value = fallback;
    if (get != NULL && to_host != NULL && get(&at, &bytes, module, name) == 0 && bytes == 4)
        to_host(&value, at, 4);
    return value;
}

void WINAPI d4rEngineNetDestroy(void* handle)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    if (net == NULL)
        return;
    HIP_P_FN synchronize = (HIP_P_FN)hip_symbol("hipStreamSynchronize");
    HIP_P_FN free_device = (HIP_P_FN)hip_symbol("hipFree");
    HIP_P_FN unload = (HIP_P_FN)hip_symbol("hipModuleUnload");
    HIP_P_FN event_destroy = (HIP_P_FN)hip_symbol("hipEventDestroy");
    HIP_P_FN stream_destroy = (HIP_P_FN)hip_symbol("hipStreamDestroy");
    /* A queued input wait whose frame never reached the GPU would make the synchronize below wait forever:
       satisfy it (the shim also releases waits it knows about, before it destroys the network). */
    if (net->input_wait != 0 && net->input_wait_value != 0)
        d4rEngineNetReleaseInput(net, net->input_wait_value);
    if (net->stream != NULL && synchronize != NULL)
        synchronize(net->stream);
    HIP_P_FN exec_destroy = (HIP_P_FN)hip_symbol("hipGraphExecDestroy");
    HIP_P_FN graph_destroy = (HIP_P_FN)hip_symbol("hipGraphDestroy");
    for (unsigned phase = 0; phase < D4R_ENGINE_NET_GRAPHS; ++phase)
    {
        if (net->executable[phase] != NULL && exec_destroy != NULL)
            exec_destroy(net->executable[phase]);
        if (net->graph[phase] != NULL && graph_destroy != NULL)
            graph_destroy(net->graph[phase]);
    }
    for (int i = 0; i < net->owned_count && free_device != NULL; ++i)
        free_device(net->owned[i]);
    for (int i = 0; i < net->module_count && unload != NULL; ++i)
        unload(net->modules[i]);
    if (net->start != NULL && event_destroy != NULL)
        event_destroy(net->start);
    if (net->stop != NULL && event_destroy != NULL)
        event_destroy(net->stop);
    if (net->stream != NULL && stream_destroy != NULL)
        stream_destroy(net->stream);
    if (net->timeline_device != NULL)
        pthread_mutex_destroy(&net->timeline_lock);
    free(net);
}

static CUresult d4r_engine_net_capture(D4rEngineNet* net, HIP_MODULE_LAUNCH_FN launch)
{
    HIP_CAPTURE_BEGIN_FN begin = (HIP_CAPTURE_BEGIN_FN)hip_symbol("hipStreamBeginCapture");
    HIP_CAPTURE_END_FN end = (HIP_CAPTURE_END_FN)hip_symbol("hipStreamEndCapture");
    HIP_GRAPH_INSTANTIATE_FN instantiate = (HIP_GRAPH_INSTANTIATE_FN)hip_symbol("hipGraphInstantiate");
    if (begin == NULL || end == NULL || instantiate == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    /* Every frame of the alignment cycle has its own immutable graph (arguments and grids frozen, the K
       bottleneck's post phases included); no per-frame node updates. */
    for (uint32_t phase = 0; phase < net->graphs; ++phase)
    {
        if (begin(net->stream, 2 /* hipStreamCaptureModeRelaxed */) != 0)
            return CUDA_ERROR_UNKNOWN;
        int queued = 0;
        for (uint32_t i = 0; i < net->node_count[phase] && queued == 0; ++i)
        {
            D4rEngineNetNode* node = &net->nodes[phase][i];
            void* args[] = {&node->params};
            queued = launch(node->function, node->gx, node->gy, 1, node->bx, node->by, node->bz, 0, net->stream, args,
                            NULL);
        }
        const int captured = end(net->stream, &net->graph[phase]);
        if (queued != 0 || captured != 0)
            return CUDA_ERROR_LAUNCH_FAILED;
        if (instantiate(&net->executable[phase], net->graph[phase], NULL, NULL, 0) != 0)
            return CUDA_ERROR_UNKNOWN;
    }
    return CUDA_SUCCESS;
}

CUresult WINAPI d4rEngineNetCreate(const uint8_t* file, uint64_t file_bytes, const D4rEngineNetCode* codes,
                                   uint32_t code_count, uint32_t width, uint32_t height, CUdeviceptr tokens,
                                   CUdeviceptr output, void** handle)
{
    ensure_context();
    if (file == NULL || codes == NULL || handle == NULL || tokens == 0 || output == 0)
        return CUDA_ERROR_INVALID_VALUE;
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    typedef int (*STREAM_PRIORITY_RANGE_FN)(int*, int*);
    typedef int (*STREAM_CREATE_PRIORITY_FN)(void**, unsigned int, int);
    STREAM_PRIORITY_RANGE_FN priority_range = (STREAM_PRIORITY_RANGE_FN)hip_symbol("hipDeviceGetStreamPriorityRange");
    STREAM_CREATE_PRIORITY_FN stream_create = (STREAM_CREATE_PRIORITY_FN)hip_symbol("hipStreamCreateWithPriority");
    HIP_PP_FN event_create = (HIP_PP_FN)hip_symbol("hipEventCreate");
    HIP_P_FN synchronize = (HIP_P_FN)hip_symbol("hipStreamSynchronize");
    HIP_MALLOC_FN allocate = (HIP_MALLOC_FN)hip_symbol("hipMalloc");
    HIP_MEMCPY_FN to_device = (HIP_MEMCPY_FN)hip_symbol("hipMemcpyHtoD");
    HIP_MODULE_LOAD_DATA_FN load = (HIP_MODULE_LOAD_DATA_FN)hip_symbol("hipModuleLoadData");
    HIP_MODULE_GET_FUNCTION_FN function = (HIP_MODULE_GET_FUNCTION_FN)hip_symbol("hipModuleGetFunction");
    HIP_MODULE_LAUNCH_FN launch = (HIP_MODULE_LAUNCH_FN)hip_symbol("hipModuleLaunchKernel");
    if (priority_range == NULL || stream_create == NULL || event_create == NULL || synchronize == NULL || allocate == NULL || to_device == NULL ||
        load == NULL || function == NULL || launch == NULL || hip_symbol("hipEventRecord") == NULL ||
        hip_symbol("hipEventQuery") == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (file_bytes < 16)
        return CUDA_ERROR_INVALID_VALUE;
    const int k_preset = memcmp(file, "D4RKHIP1", 8) == 0;
    if (!k_preset && memcmp(file, "D4RMHIP1", 8) != 0)
        return CUDA_ERROR_INVALID_VALUE;
    uint32_t count = 0;
    memcpy(&count, file + 8, 4);
    const uint32_t layers = k_preset ? D4R_ENGINE_NET_K_LAYERS : D4R_ENGINE_NET_M_LAYERS;
    if (count != layers || file_bytes < 16 + 64ull * count)
        return CUDA_ERROR_INVALID_VALUE;
    int least_priority = 0, greatest_priority = 0;
    if (priority_range(&least_priority, &greatest_priority) != 0)
        return CUDA_ERROR_UNKNOWN;
    D4rEngineNet* net = (D4rEngineNet*)calloc(1, sizeof(*net));
    if (net == NULL)
        return CUDA_ERROR_OUT_OF_MEMORY;
    net->k_preset = k_preset;
    net->graphs = k_preset ? D4R_ENGINE_NET_GRAPHS : 2;
    const char* failure = NULL;
    /* The layer-0 token grid: M's render dimensions, K's output dimensions (makeKShape). */
    uint32_t tw, th;
    if (k_preset)
    {
        if (width < 256 || height < 256 || width > 8192 || height > 8192)
            failure = "output dimensions";
        tw = ((width + 3) / 4 + 31) / 32 * 32;
        th = ((height + 3) / 4 + 31) / 32 * 32;
        if (tw < 256) tw = 256;
        if (th < 256) th = 256;
    }
    else
    {
        tw = (width + 31) / 32 * 16;
        th = (height + 31) / 32 * 16;
    }
    /* This stream is on the frame's critical path, between two parts of the game's graphics list. */
    if (failure == NULL && (stream_create(&net->stream, 1 /* hipStreamNonBlocking */, greatest_priority) != 0 ||
                            event_create(&net->start) != 0 || event_create(&net->stop) != 0))
        failure = "stream or events";
    void* const tok = (void*)(uintptr_t)tokens;
    void* const result = (void*)(uintptr_t)output;
    D4rEngineNetKernel kern[D4R_ENGINE_NET_MODULES];
    memset(kern, 0, sizeof(kern));
    int module_count = 0;
    void* layer_in[D4R_ENGINE_NET_K_LAYERS] = {NULL};
    void* layer_skip[D4R_ENGINE_NET_K_LAYERS] = {NULL};
    void* layer_out[D4R_ENGINE_NET_K_LAYERS] = {NULL};
    void* layer_merged[D4R_ENGINE_NET_K_LAYERS] = {NULL};
    void* k_full[5] = {NULL};
    void* weights[D4R_ENGINE_NET_K_LAYERS] = {NULL};
    uint32_t divisor[D4R_ENGINE_NET_K_LAYERS] = {0};
    int32_t shift_sx[D4R_ENGINE_NET_K_LAYERS] = {0}, shift_sy[D4R_ENGINE_NET_K_LAYERS] = {0};
    uint32_t layer_kernel[D4R_ENGINE_NET_K_LAYERS] = {0};
    D4rEngineNetParamsU layer_params[D4R_ENGINE_NET_K_LAYERS];
    memset(layer_params, 0, sizeof(layer_params));
    if (k_preset && failure == NULL)
    {
        /* NGX packs the encoder skips, the merged outputs and the decoder outputs into one scratch allocation;
           separate buffers only have to respect the channel counts and the wiring, not the original aliasing. */
        void* act[D4R_ENGINE_NET_K_LAYERS] = {NULL};
        for (uint32_t i = 0; i < layers && failure == NULL; ++i)
        {
            const D4rEngineNetKLayer* p = &d4r_engine_net_k_plan[i];
            const uint32_t w = tw / p->divisor, h = th / p->divisor;
            /* an encoder's out24 is the patch-merged 2x2 output, half the resolution of its input */
            const uint32_t ow = i < 5 ? tw / (2 * p->divisor) : w, oh = i < 5 ? th / (2 * p->divisor) : h;
            const int need = (i + 1 != layers ? 1 : 0) + (i < 5 ? 1 : 0);
            if (net->owned_count + need > D4R_ENGINE_NET_OWNED)
            {
                failure = "activation buffers";
                break;
            }
            /* dec0 writes the head straight into the caller's output buffer */
            if (i + 1 != layers)
            {
                if (allocate(&act[i], (size_t)ow * oh * p->out24ch * 2 + 4096) != 0)
                {
                    failure = "activation buffers";
                    break;
                }
                net->owned[net->owned_count++] = act[i];
            }
            if (i < 5)
            {
                if (allocate(&k_full[i], (size_t)w * h * p->c * 2 + 4096) != 0)
                {
                    failure = "activation buffers";
                    break;
                }
                net->owned[net->owned_count++] = k_full[i];
            }
            layer_in[i] = i == 0 ? tok : act[i - 1];
            layer_skip[i] = i >= 6 ? k_full[10 - i] : NULL;
            layer_out[i] = i + 1 == layers ? result : act[i];
        }
    }
    else if (failure == NULL)
    {
        /* skip1, merged1, skip2, merged2, tubeA, tubeB, dec2 */
        static const uint32_t scale[7] = {1, 2, 2, 4, 4, 4, 2}, channels[7] = {64, 96, 96, 128, 128, 128, 96};
        void* b[7] = {NULL};
        for (int i = 0; i < 7 && failure == NULL; ++i)
        {
            if (net->owned_count >= D4R_ENGINE_NET_OWNED ||
                allocate(&b[i], (size_t)(tw / scale[i]) * (th / scale[i]) * channels[i] + 4096) != 0)
                failure = "activation buffers";
            else
                net->owned[net->owned_count++] = b[i];
        }
        /* tokens -> enc1 -> enc2 -> six tube blocks (ping-pong) -> dec2 -> dec1 */
        const int in_i[10] = {0, 1, 3, 4, 5, 4, 5, 4, 5, 6};
        const int skip_i[10] = {-1, -1, -1, -1, -1, -1, -1, -1, 2, 0};
        const int out_i[10] = {0, 2, 4, 5, 4, 5, 4, 5, 6, -1};
        const int merged_i[10] = {1, 3, -1, -1, -1, -1, -1, -1, -1, -1};
        for (int i = 0; i < 10; ++i)
        {
            layer_in[i] = in_i[i] < 0 ? NULL : b[in_i[i]];
            layer_skip[i] = skip_i[i] < 0 ? NULL : b[skip_i[i]];
            layer_out[i] = out_i[i] < 0 ? result : b[out_i[i]];
            layer_merged[i] = merged_i[i] < 0 ? NULL : b[merged_i[i]];
        }
        layer_in[0] = tok; /* the engine's token buffer feeds the first layer, not a scratch buffer */
    }
    for (uint32_t i = 0; i < count && failure == NULL; ++i)
    {
        const uint8_t* e = file + 16 + 64 * i;
        char name[33] = {0};
        int32_t sx, sy;
        uint32_t div;
        uint64_t at, bytes;
        memcpy(name, e, 32);
        memcpy(&sx, e + 32, 4);
        memcpy(&sy, e + 36, 4);
        memcpy(&div, e + 40, 4);
        memcpy(&at, e + 48, 8);
        memcpy(&bytes, e + 56, 8);
        if (div == 0 || at > file_bytes || bytes > file_bytes - at)
        {
            failure = "layer table";
            break;
        }
        if (k_preset)
        {
            char expect[48];
            snprintf(expect, sizeof(expect), "dltss_pwin_%s_layer", d4r_engine_net_k_plan[i].name);
            if (strcmp(name, expect) != 0)
            {
                failure = "layer table";
                break;
            }
        }
        int k = 0;
        while (k < module_count && strcmp(kern[k].name, name) != 0)
            ++k;
        if (k == module_count)
        {
            const D4rEngineNetCode* code = NULL;
            for (uint32_t c = 0; c < code_count; ++c)
                if (codes[c].name != NULL && strcmp(codes[c].name, name) == 0)
                    code = &codes[c];
            if (code == NULL || k == D4R_ENGINE_NET_MODULES || load(&net->modules[k], code->data) != 0)
            {
                failure = "kernel code object";
                break;
            }
            ++net->module_count;
            ++module_count;
            strcpy(kern[k].name, name);
            kern[k].module = net->modules[k];
            char prep_name[48];
            snprintf(prep_name, sizeof(prep_name), "%s_prep", name);
            if (function(&kern[k].main, kern[k].module, name) != 0)
            {
                failure = "kernel entry point";
                break;
            }
            if (function(&kern[k].prep, kern[k].module, prep_name) != 0)
                kern[k].prep = NULL;
            kern[k].prep_blocks = d4r_engine_net_global(kern[k].module, "d4r_prep_blocks", 0);
            kern[k].block_z = d4r_engine_net_global(kern[k].module, "d4r_block_z", 4);
            kern[k].grid_x = d4r_engine_net_global(kern[k].module, "d4r_grid_x", 0);
            kern[k].grid_y = d4r_engine_net_global(kern[k].module, "d4r_grid_y", 0);
            if (kern[k].block_z == 0)
                kern[k].block_z = 4;
            /* one prepared weight image per weights pointer while a module's slots last: the tube needs six */
            if ((kern[k].prep != NULL && kern[k].prep_blocks == 0) ||
                (strstr(name, "tube") != NULL && d4r_engine_net_global(kern[k].module, "d4r_prep_key_slots", 0) < 6))
            {
                failure = "weight image slots";
                break;
            }
            /* the deep layers' code objects also carry their post phases; their grids are compile-time */
            for (int j = 1; j <= 8; ++j)
            {
                char tag[48], sym[48];
                snprintf(tag, sizeof(tag), "%s_post%d", name, j);
                if (function(&kern[k].post[kern[k].post_count], kern[k].module, tag) != 0)
                    break;
                snprintf(sym, sizeof(sym), "d4r_post%d_grid_x", j);
                kern[k].post_gx[kern[k].post_count] = d4r_engine_net_global(kern[k].module, sym, 0);
                snprintf(sym, sizeof(sym), "d4r_post%d_grid_y", j);
                kern[k].post_gy[kern[k].post_count] = d4r_engine_net_global(kern[k].module, sym, 0);
                snprintf(sym, sizeof(sym), "d4r_post%d_block_x", j);
                kern[k].post_bx[kern[k].post_count] = d4r_engine_net_global(kern[k].module, sym, 0);
                snprintf(sym, sizeof(sym), "d4r_post%d_block_y", j);
                kern[k].post_by[kern[k].post_count] = d4r_engine_net_global(kern[k].module, sym, 0);
                snprintf(sym, sizeof(sym), "d4r_post%d_block_z", j);
                kern[k].post_bz[kern[k].post_count] = d4r_engine_net_global(kern[k].module, sym, 0);
                if (kern[k].post_gx[kern[k].post_count] == 0)
                    kern[k].post_gx[kern[k].post_count] = kern[k].grid_x;
                if (kern[k].post_bx[kern[k].post_count] == 0)
                    kern[k].post_bx[kern[k].post_count] = 32;
                if (kern[k].post_by[kern[k].post_count] == 0)
                    kern[k].post_by[kern[k].post_count] = 1;
                if (kern[k].post_bz[kern[k].post_count] == 0)
                    kern[k].post_bz[kern[k].post_count] = 1;
                ++kern[k].post_count;
            }
        }
        void* w = NULL;
        if (net->owned_count >= D4R_ENGINE_NET_OWNED || allocate(&w, bytes + 4096) != 0)
        {
            failure = "weights";
            break;
        }
        net->owned[net->owned_count++] = w;
        if (to_device(w, file + at, bytes) != 0)
        {
            failure = "weights upload";
            break;
        }
        weights[i] = w;
        divisor[i] = div;
        shift_sx[i] = sx;
        shift_sy[i] = sy;
        layer_kernel[i] = (uint32_t)k;
    }
    /* The frozen parameter block of every layer at phase/frame 0; the graph loop copies and shifts it. */
    for (uint32_t i = 0; i < count && failure == NULL; ++i)
    {
        if (k_preset)
        {
            D4rEngineNetPwinParams* p = &layer_params[i].k;
            p->W = (int32_t)(tw / divisor[i]);
            p->H = (int32_t)(th / divisor[i]);
            p->in = (uint8_t*)layer_in[i];
            p->skip = (uint8_t*)layer_skip[i];
            p->out24 = (uint8_t*)layer_out[i];
            p->out32 = i < 5 ? (uint8_t*)k_full[i] : NULL;
            p->w = (uint8_t*)weights[i];
            d4r_engine_net_k_shift(d4r_engine_net_k_plan[i].role, 0, &p->sx, &p->sy);
        }
        else
        {
            D4rEngineNetParams* p = &layer_params[i].m;
            p->w = weights[i];
            p->sx = shift_sx[i];
            p->sy = shift_sy[i];
            p->tw = (int32_t)(tw / divisor[i]);
            p->th = (int32_t)(th / divisor[i]);
            p->in = layer_in[i];
            p->skip = layer_skip[i];
            p->out = layer_out[i];
            p->merged = layer_merged[i];
        }
    }
    /* Every frame of the alignment cycle has its own immutable graph: frozen grids, blocks and arguments,
       the K bottleneck's post phases included. No per-frame allocation or parameter copy. */
    for (uint32_t phase = 0; phase < net->graphs && failure == NULL; ++phase)
    {
        uint32_t n = 0;
        for (uint32_t i = 0; i < count && failure == NULL; ++i)
        {
            const D4rEngineNetKernel* kk = &kern[layer_kernel[i]];
            if (n >= D4R_ENGINE_NET_NODES)
            {
                failure = "too many nodes";
                break;
            }
            D4rEngineNetNode* node = &net->nodes[phase][n++];
            node->function = kk->main;
            node->bx = 32;
            node->by = 1;
            node->bz = kk->block_z;
            node->params = layer_params[i];
            if (k_preset)
            {
                D4rEngineNetPwinParams* p = &node->params.k;
                d4r_engine_net_k_shift(d4r_engine_net_k_plan[i].role, phase, &p->sx, &p->sy);
                node->gx = kk->post_count != 0 ? (kk->grid_x != 0 ? kk->grid_x : 32)
                                               : (uint32_t)(p->W + p->sx + 7) / 8;
                node->gy = kk->post_count != 0 ? (kk->grid_y != 0 ? kk->grid_y : 1)
                                               : (uint32_t)(p->H + p->sy + 7) / 8;
                for (int j = 0; j < kk->post_count && failure == NULL; ++j)
                {
                    if (n >= D4R_ENGINE_NET_NODES)
                    {
                        failure = "too many nodes";
                        break;
                    }
                    D4rEngineNetNode* post = &net->nodes[phase][n++];
                    post->function = kk->post[j];
                    post->params = node->params; /* the same PwinParams block */
                    post->gx = kk->post_gx[j];
                    post->gy = kk->post_gy[j] != 0 ? kk->post_gy[j] : 1;
                    post->bx = kk->post_bx[j];
                    post->by = kk->post_by[j];
                    post->bz = kk->post_bz[j];
                }
            }
            else
            {
                D4rEngineNetParams* p = &node->params.m;
                /* NGX alternates horizontal window alignment every evaluation, including reset evaluations. */
                p->sx = shift_sx[i] ^ (int32_t)(phase * 2);
                node->gx = (uint32_t)(p->tw + p->sx + 7) / 8;
                node->gy = (uint32_t)(p->th + shift_sy[i] + 7) / 8;
            }
        }
        net->node_count[phase] = n;
    }
    /* The weights never change: every layer's weight image is prepared once, here, and found by its pointer.
       A shared code object (the M tube) gets one preparation per layer, i.e. per distinct weights pointer. */
    for (uint32_t i = 0; i < count && failure == NULL; ++i)
    {
        const D4rEngineNetKernel* kk = &kern[layer_kernel[i]];
        if (kk->prep == NULL)
            continue;
        if (net->prep_count >= D4R_ENGINE_NET_LAYERS)
        {
            failure = "weight image slots";
            break;
        }
        D4rEngineNetNode* prep = &net->preps[net->prep_count++];
        prep->function = kk->prep;
        prep->gx = kk->prep_blocks;
        prep->bx = 128;
        prep->by = 1;
        prep->bz = 1;
        prep->params = layer_params[i];
    }
    for (uint32_t i = 0; i < net->prep_count && failure == NULL; ++i)
    {
        D4rEngineNetNode* prep = &net->preps[i];
        void* args[] = {&prep->params};
        if (launch(prep->function, prep->gx, 1, 1, prep->bx, prep->by, prep->bz, 0, net->stream, args, NULL) != 0)
            failure = "weight preparation";
    }
    if (failure == NULL && synchronize(net->stream) != 0)
        failure = "weight preparation";
    if (failure == NULL && d4r_engine_net_capture(net, launch) != CUDA_SUCCESS)
        failure = "network graph capture";
    if (failure == NULL)
    {
        HIP_EVENT_RECORD_FN upload = (HIP_EVENT_RECORD_FN)hip_symbol("hipGraphUpload");
        if (upload == NULL)
            failure = "network graph upload";
        for (uint32_t phase = 0; phase < net->graphs && failure == NULL; ++phase)
            if (upload(net->executable[phase], net->stream) != 0)
                failure = "network graph upload";
        if (failure == NULL && synchronize(net->stream) != 0)
            failure = "network graph upload";
    }
    if (failure != NULL)
    {
        tracef("d4rEngineNetCreate: %s failed", failure);
        d4rEngineNetDestroy(net);
        return CUDA_ERROR_UNKNOWN;
    }
    *handle = net;
    tracef("d4rEngineNetCreate: preset %s network on native HIP layers, %ux%u tokens, %u graphs", k_preset ? "K" : "M",
           tw, th, net->graphs);
    return CUDA_SUCCESS;
}

/* All writers (including cancellation retirement in the shim) use this lock. The callback runs on a native
   HIP host thread: no Wine/PE entry points and no HIP APIs are called from it. */
int WINAPI d4rEngineNetSignalTimeline(void* handle, uint64_t value)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    if (net == NULL || net->timeline_device == NULL)
        return -13; /* VK_ERROR_UNKNOWN */
    pthread_mutex_lock(&net->timeline_lock);
    int result = 0;
    if (value > net->timeline_value)
    {
        D4rVkSemaphoreSignalInfo info = {1000207005, NULL, net->timeline_semaphore, value};
        result = net->signal_timeline(net->timeline_device, &info);
        if (result == 0)
            net->timeline_value = value;
    }
    pthread_mutex_unlock(&net->timeline_lock);
    return result;
}

static void d4r_engine_net_completed(void* data)
{
    D4rEngineNetSignal* signal = (D4rEngineNetSignal*)data;
    const int result = d4rEngineNetSignalTimeline(signal->net, signal->value);
    if (result != 0)
        __atomic_store_n(&signal->net->timeline_error, result, __ATOMIC_RELEASE);
    __atomic_store_n(&signal->busy, 0, __ATOMIC_RELEASE);
}

CUresult WINAPI d4rEngineNetSetTimeline(void* handle, void* client_device, uint64_t client_semaphore)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    if (net == NULL || net->timeline_device != NULL || client_device == NULL || client_semaphore == 0)
        return CUDA_ERROR_INVALID_VALUE;
    const uint64_t* device_object = (const uint64_t*)(uintptr_t)((const uint64_t*)client_device)[1];
    const uint64_t* semaphore_object = (const uint64_t*)(uintptr_t)client_semaphore;
    if (device_object == NULL || device_object[1] != (uint64_t)(uintptr_t)client_device ||
        semaphore_object[1] != client_semaphore)
        return CUDA_ERROR_NOT_SUPPORTED;
    void* host_device = (void*)(uintptr_t)device_object[0];
    void* vulkan = dlopen("libvulkan.so.1", RTLD_NOW | RTLD_NOLOAD);
    VK_GET_DEVICE_PROC_ADDR_FN get_proc =
        vulkan != NULL ? (VK_GET_DEVICE_PROC_ADDR_FN)dlsym(vulkan, "vkGetDeviceProcAddr") : NULL;
    HIP_LAUNCH_HOST_FN launch_host = (HIP_LAUNCH_HOST_FN)hip_symbol("hipLaunchHostFunc");
    if (get_proc == NULL || launch_host == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    VK_SIGNAL_SEMAPHORE_FN signal = (VK_SIGNAL_SEMAPHORE_FN)get_proc(host_device, "vkSignalSemaphore");
    VK_GET_SEMAPHORE_COUNTER_FN counter = (VK_GET_SEMAPHORE_COUNTER_FN)get_proc(host_device, "vkGetSemaphoreCounterValue");
    uint64_t value = 0;
    if (signal == NULL || counter == NULL || counter(host_device, semaphore_object[0], &value) != 0 ||
        pthread_mutex_init(&net->timeline_lock, NULL) != 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    net->timeline_device = host_device;
    net->timeline_semaphore = semaphore_object[0];
    net->timeline_value = value;
    net->signal_timeline = signal;
    net->launch_host = launch_host;
    for (unsigned i = 0; i < D4R_ENGINE_NET_SIGNALS; ++i)
        net->signals[i].net = net;
    return CUDA_SUCCESS;
}

/* Reserve before queuing this frame's input wait. A burst of discarded lists can leave callbacks in flight;
   draining the previous work is safe here, but would deadlock on a newly queued wait for an unsubmitted list. */
static D4rEngineNetSignal* d4r_engine_net_signal_slot(D4rEngineNet* net, uint32_t value)
{
    if (net->timeline_device == NULL || value == 0)
        return NULL;
    for (unsigned pass = 0; pass < 2; ++pass)
    {
        for (unsigned i = 0; i < D4R_ENGINE_NET_SIGNALS; ++i)
        {
            int free = 0;
            if (__atomic_compare_exchange_n(&net->signals[i].busy, &free, 1, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED))
            {
                net->signals[i].value = value;
                return &net->signals[i];
            }
        }
        HIP_P_FN synchronize = (HIP_P_FN)hip_symbol("hipStreamSynchronize");
        if (pass != 0 || synchronize == NULL || synchronize(net->stream) != 0)
            break;
    }
    return NULL;
}

static CUresult d4r_engine_net_queue(D4rEngineNet* net, HIP_EVENT_RECORD_FN graph_launch, HIP_EVENT_RECORD_FN record,
                                    D4rEngineNetSignal* signal)
{
    CUresult result = CUDA_ERROR_UNKNOWN;
    if (record(net->start, net->stream) == 0)
    {
        /* One evaluation advances the alignment cycle by one frame; a reset does not restart it. */
        result = graph_launch(net->executable[net->frame % net->graphs], net->stream) == 0
                     ? CUDA_SUCCESS : CUDA_ERROR_LAUNCH_FAILED;
        if (result == CUDA_SUCCESS)
        {
            ++net->frame;
            if (record(net->stop, net->stream) != 0 ||
                net->launch_host(net->stream, d4r_engine_net_completed, signal) != 0)
                result = CUDA_ERROR_UNKNOWN;
        }
    }
    if (result != CUDA_SUCCESS)
        __atomic_store_n(&signal->busy, 0, __ATOMIC_RELEASE);
    return result;
}

CUresult WINAPI d4rEngineNetLaunch(void* handle, uint32_t value)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    HIP_EVENT_RECORD_FN graph_launch = (HIP_EVENT_RECORD_FN)hip_symbol("hipGraphLaunch");
    HIP_EVENT_RECORD_FN record = (HIP_EVENT_RECORD_FN)hip_symbol("hipEventRecord");
    if (net == NULL || graph_launch == NULL || record == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    D4rEngineNetSignal* signal = d4r_engine_net_signal_slot(net, value);
    if (signal == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    return d4r_engine_net_queue(net, graph_launch, record, signal);
}

/* Check HIP's wait-value capability and a round trip on this imported buffer before enabling LaunchAt.
   Release the probe wait before querying its event: HIP may flush the blocked queue during a query.
   The test leaves the word at 0; frame values then increase monotonically and use a Gte wait. */
int WINAPI d4rEngineNetInputWaitReady(void* handle, CUdeviceptr sync, uint32_t* host)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    HIP_STREAM_WAIT_VALUE32_FN wait = (HIP_STREAM_WAIT_VALUE32_FN)hip_symbol("hipStreamWaitValue32");
    HIP_STREAM_WRITE_VALUE32_FN write = (HIP_STREAM_WRITE_VALUE32_FN)hip_symbol("hipStreamWriteValue32");
    HIP_EVENT_RECORD_FN record = (HIP_EVENT_RECORD_FN)hip_symbol("hipEventRecord");
    HIP_P_FN query = (HIP_P_FN)hip_symbol("hipEventQuery");
    HIP_PP_FN event_create = (HIP_PP_FN)hip_symbol("hipEventCreate");
    HIP_P_FN event_destroy = (HIP_P_FN)hip_symbol("hipEventDestroy");
    HIP_P_FN synchronize = (HIP_P_FN)hip_symbol("hipStreamSynchronize");
    HIP_MEMCPY_FN to_host = (HIP_MEMCPY_FN)hip_symbol("hipMemcpyDtoH");
    typedef int (*GET_ATTRIBUTE_FN)(int*, int, int);
    typedef int (*GET_DEVICE_FN)(int*);
    GET_ATTRIBUTE_FN attribute = (GET_ATTRIBUTE_FN)hip_symbol("hipDeviceGetAttribute");
    GET_DEVICE_FN device = (GET_DEVICE_FN)hip_symbol("hipGetDevice");
    int current = 0, supported = 0;
    if (attribute == NULL || device == NULL || device(&current) != 0 ||
        attribute(&supported, 10013 /* hipDeviceAttributeCanUseStreamWaitValue */, current) != 0 || !supported)
        return 0;
    if (net == NULL || sync == 0 || host == NULL || wait == NULL || write == NULL || record == NULL || query == NULL ||
        event_create == NULL || event_destroy == NULL || synchronize == NULL || to_host == NULL)
        return 0;
    /* Set for the probe's own releases; cleared again below unless the probe proves GPU-side waits work. */
    net->input_wait = sync;
    net->input_wait_host = host;
    void* probe = NULL;
    if (event_create(&probe) != 0)
    {
        net->input_wait = 0;
        net->input_wait_host = NULL;
        return 0;
    }
    /* HIP lazily initializes queue/event storage. Do that before blocking a queue: allocating it after the
       wait has been submitted can itself synchronize against the blocked queue. */
    if (d4rEngineNetReleaseInput(net, 0) != CUDA_SUCCESS || record(probe, net->stream) != 0 ||
        record(net->start, net->stream) != 0 || record(net->stop, net->stream) != 0 ||
        synchronize(net->stream) != 0)
    {
        event_destroy(probe);
        net->input_wait = 0;
        net->input_wait_host = NULL;
        return 0;
    }
    int ready = 0, wedged = 0;
    if (write(net->stream, (void*)(uintptr_t)sync, 0, 0) == 0 && synchronize(net->stream) == 0 &&
        wait(net->stream, (void*)(uintptr_t)sync, 1, 0 /* hipStreamWaitValueGte */, 0xffffffffu) == 0)
    {
        /* The wait is queued now: it must be satisfied before the stream is synchronized again. */
        const int recorded = record(probe, net->stream);
        int state = 600; /* hipErrorNotReady */
        if (d4rEngineNetReleaseInput(net, 1) != CUDA_SUCCESS)
            wedged = 1;
        else if (recorded == 0)
        {
            for (int i = 0; i < 4000 && state == 600; ++i)
            {
                usleep(50);
                state = query(probe);
            }
            if (state == 600)
                wedged = 1;
            else
                ready = state == 0;
        }
        if (!wedged)
        {
            /* Back to 0 for the frames' own values, with no wait left queued. */
            if (write(net->stream, (void*)(uintptr_t)sync, 0, 0) != 0 || synchronize(net->stream) != 0)
                ready = 0;
            if (ready != 0)
            {
                uint32_t value = 0;
                if (to_host(&value, (void*)(uintptr_t)sync, 4) != 0 || value != 0)
                    ready = 0;
            }
        }
    }
    event_destroy(probe);
    if (wedged)
    {
        /* Destroy releases it (d4rEngineNetDestroy); the caller must not use this network. */
        net->input_wait = sync;
        net->input_wait_value = 1;
    }
    else if (!ready)
    {
        /* d4rEngineNetLaunchAt refuses without a proven wait. */
        net->input_wait = 0;
        net->input_wait_host = NULL;
    }
    tracef("d4rEngineNetInputWaitReady(%p): %s", (void*)(uintptr_t)sync,
           wedged ? "the stream is stuck" : ready ? "GPU-side waits" : "not supported");
    return wedged ? -1 : ready;
}

/* Queues a wait for `value` (the frame number the engine's front fills into the sync buffer) before this
   evaluation's graph, so the network starts on the GPU the moment its inputs are complete. */
CUresult WINAPI d4rEngineNetLaunchAt(void* handle, uint32_t value)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    HIP_EVENT_RECORD_FN graph_launch = (HIP_EVENT_RECORD_FN)hip_symbol("hipGraphLaunch");
    HIP_EVENT_RECORD_FN record = (HIP_EVENT_RECORD_FN)hip_symbol("hipEventRecord");
    HIP_STREAM_WAIT_VALUE32_FN wait = (HIP_STREAM_WAIT_VALUE32_FN)hip_symbol("hipStreamWaitValue32");
    if (net == NULL || graph_launch == NULL || record == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    if (net->input_wait == 0 || wait == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    D4rEngineNetSignal* signal = d4r_engine_net_signal_slot(net, value);
    if (signal == NULL)
        return CUDA_ERROR_INVALID_VALUE;
    if (wait(net->stream, (void*)(uintptr_t)net->input_wait, value, 0 /* hipStreamWaitValueGte */, 0xffffffffu) != 0)
    {
        __atomic_store_n(&signal->busy, 0, __ATOMIC_RELEASE);
        return CUDA_ERROR_NOT_SUPPORTED;
    }
    net->input_wait_value = value;
    // Keep the queued wait tracked even if a later event/kernel enqueue fails; destroy must release it.
    return d4r_engine_net_queue(net, graph_launch, record, signal);
}

/* 1: the launched layers have finished; 0: still running; negative: error */
int WINAPI d4rEngineNetDone(void* handle, float* milliseconds)
{
    D4rEngineNet* net = (D4rEngineNet*)handle;
    HIP_P_FN query = (HIP_P_FN)hip_symbol("hipEventQuery");
    HIP_EVENT_ELAPSED_FN elapsed = (HIP_EVENT_ELAPSED_FN)hip_symbol("hipEventElapsedTime");
    if (net == NULL || query == NULL || __atomic_load_n(&net->timeline_error, __ATOMIC_ACQUIRE) != 0)
        return -1;
    const int result = query(net->stop);
    if (result == 600 /* hipErrorNotReady */)
        return 0;
    if (result != 0)
        return -1;
    /* The layers ran, so the input wait (if any) is behind us. */
    net->input_wait_value = 0;
    if (milliseconds != NULL && (elapsed == NULL || elapsed(milliseconds, net->start, net->stop) != 0))
        *milliseconds = 0.0f;
    return 1;
}

/* d4r: asynchronous 2D copy between device memory and arrays on the null stream (ZLUDA forwards
   CUarray handles as hipArray_t and the null CUstream as HIP's null stream; ZLUDA itself has no
   cuMemcpy2DAsync). The CUDA and HIP descriptors share their layout; only the memory type codes
   differ. Other streams and host memory are rejected so callers fall back to cuMemcpy2D. */
typedef int (*HIP_MEMCPY_PARAM_2D_ASYNC_FN)(const D4rMemcpy2D*, void*);

static int d4r_hip_memory_type(unsigned int cuda_type)
{
    return cuda_type == 2 ? 2 /* hipMemoryTypeDevice */ : cuda_type == 3 ? 10 /* hipMemoryTypeArray */ : -1;
}

CUresult WINAPI d4rMemcpy2DAsync(const D4rMemcpy2D* copy, CUstream stream)
{
    static HIP_MEMCPY_PARAM_2D_ASYNC_FN function;
    ensure_context();
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    if (copy == NULL || stream != NULL)
        return CUDA_ERROR_INVALID_VALUE;
    const int src = d4r_hip_memory_type(copy->srcMemoryType), dst = d4r_hip_memory_type(copy->dstMemoryType);
    if (src < 0 || dst < 0)
        return CUDA_ERROR_NOT_SUPPORTED;
    if (function == NULL)
        function = (HIP_MEMCPY_PARAM_2D_ASYNC_FN)hip_symbol("hipMemcpyParam2DAsync");
    if (function == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    D4rMemcpy2D hip = *copy;
    hip.srcMemoryType = (unsigned int)src;
    hip.dstMemoryType = (unsigned int)dst;
    const int result = function(&hip, NULL);
    if (result != 0 || trace_verbose())
        tracef("d4rMemcpy2DAsync %zux%zu result=%d", copy->WidthInBytes, copy->Height, result);
    return result == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

/* d4r: GPU-side handoff from the game's queue. The null stream (where NGX runs) waits until the u32 at
   `pointer` (device memory the game's queue writes with vkCmdFillBuffer) is >= value. d4rWriteValue32
   writes such a value from a separate non-blocking stream (initialisation, or releasing a stuck wait). */
CUresult WINAPI d4rStreamWaitValue32(CUdeviceptr pointer, uint32_t value)
{
    static HIP_STREAM_WAIT_VALUE32_FN function;
    ensure_context();
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    if (function == NULL)
        function = (HIP_STREAM_WAIT_VALUE32_FN)hip_symbol("hipStreamWaitValue32");
    if (function == NULL)
        return CUDA_ERROR_NOT_SUPPORTED;
    const int result = function(NULL, (void*)(uintptr_t)pointer, value, 0 /* hipStreamWaitValueGte */, 0xffffffffu);
    if (result != 0 || trace_verbose())
        tracef("d4rStreamWaitValue32 %p >= %u result=%d", (void*)(uintptr_t)pointer, value, result);
    return result == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}

CUresult WINAPI d4rWriteValue32(CUdeviceptr pointer, uint32_t value)
{
    static HIP_STREAM_WRITE_VALUE32_FN write;
    static HIP_STREAM_SYNCHRONIZE_FN sync;
    static void* stream;
    static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    ensure_context();
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    pthread_mutex_lock(&lock);
    if (stream == NULL)
    {
        HIP_STREAM_CREATE_WITH_FLAGS_FN create = (HIP_STREAM_CREATE_WITH_FLAGS_FN)hip_symbol("hipStreamCreateWithFlags");
        write = (HIP_STREAM_WRITE_VALUE32_FN)hip_symbol("hipStreamWriteValue32");
        sync = (HIP_STREAM_SYNCHRONIZE_FN)hip_symbol("hipStreamSynchronize");
        if (create == NULL || write == NULL || sync == NULL || create(&stream, 1 /* hipStreamNonBlocking */) != 0)
            stream = NULL;
    }
    int result = stream != NULL ? write(stream, (void*)(uintptr_t)pointer, value, 0) : -1;
    if (result == 0)
        result = sync(stream);
    pthread_mutex_unlock(&lock);
    tracef("d4rWriteValue32 %p = %u result=%d", (void*)(uintptr_t)pointer, value, result);
    return result == 0 ? CUDA_SUCCESS : CUDA_ERROR_NOT_SUPPORTED;
}

/* Queues (null stream) the redirect of every surface on `array` to linear memory (pitch 0: off). */
CUresult WINAPI d4rSetArrayRedirect(CUarray array, CUdeviceptr pointer, uint32_t pitch)
{
    ensure_context();
    if (context_setup_result != CUDA_SUCCESS)
        return context_setup_result;
    CUsurfObject objects[16];
    int count = 0;
    pthread_mutex_lock(&instrumentation_lock);
    redirect_array = array;
    redirect_pointer = pointer;
    redirect_pitch = pitch;
    for (SurfaceObjectRecord* current = surface_objects; current != NULL && count < 16; current = current->next)
        if (current->array == array)
            objects[count++] = current->object;
    pthread_mutex_unlock(&instrumentation_lock);
    CUresult result = CUDA_SUCCESS;
    for (int index = 0; index < count && result == CUDA_SUCCESS; ++index)
        result = write_redirect_tail(objects[index], pointer, pitch, 1);
    if (result != CUDA_SUCCESS || trace_verbose())
        tracef("d4rSetArrayRedirect array=%p -> 0x%llx pitch %u: %d surfaces, result=%d", array,
               (unsigned long long)pointer, pitch, count, result);
    return result;
}

CUresult WINAPI d4rReleaseVulkanMemory(void* memory)
{
    HIP_DESTROY_EXTERNAL_MEMORY_FN destroy = (HIP_DESTROY_EXTERNAL_MEMORY_FN)hip_symbol("hipDestroyExternalMemory");
    const int result = destroy != NULL ? destroy(memory) : -1;
    tracef("d4rReleaseVulkanMemory: hipDestroyExternalMemory(%p) -> %d", memory, result);
    return result == 0 ? CUDA_SUCCESS : CUDA_ERROR_INVALID_VALUE;
}
