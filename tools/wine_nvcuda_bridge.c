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
    CUDA_ERROR_NOT_SUPPORTED = 801,
};

static void* cuda_library;
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

    tracef("created ZLUDA context for device %d", device);
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
    case 80: return 16; /* packed 10:10:10:2 backed by RGBA32F */
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
    CUCTX_SYNCHRONIZE_FN function = (CUCTX_SYNCHRONIZE_FN)find_zluda_symbol("cuCtxSynchronize");
    CUresult result = function != NULL ? function() : missing("cuCtxSynchronize");
    if (result == CUDA_SUCCESS)
        flush_kernel_profiles();
    TRACE_CALL(result, "d4rCtxSynchronize result=%d", result);
    return result;
}

CUresult WINAPI cuCtxSynchronize(void)
{
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
    CUMODULEUNLOAD_FN function = (CUMODULEUNLOAD_FN)find_zluda_symbol("cuModuleUnload");
    CUresult result = function != NULL ? function(module) : missing("cuModuleUnload");
    tracef("cuModuleUnload module=%p result=%d", module, result);
    return result;
}

CUresult WINAPI cuModuleGetFunction(CUfunction* function_out, CUmodule module, const char* name)
{
    CUMODULEGETFUNCTION_FN function = (CUMODULEGETFUNCTION_FN)find_zluda_symbol("cuModuleGetFunction");
    CUresult result = function != NULL ? function(function_out, module, name) : missing("cuModuleGetFunction");
    if (result == CUDA_SUCCESS && function_out != NULL)
    {
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
    CUresult result = function != NULL
        ? function(function_handle, grid_x, grid_y, grid_z, block_x, block_y, block_z,
                   shared_bytes, stream, kernel_params, extra)
        : missing("cuLaunchKernel");
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
        remember_allocation(*pointer, bytes);
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
    CUMEMFREE_FN function = (CUMEMFREE_FN)find_zluda_symbol("cuMemFree_v2");
    CUresult result = function != NULL ? function(pointer) : missing("cuMemFree_v2");
    if (result == CUDA_SUCCESS)
        forget_allocation(pointer);
    TRACE_CALL(result, "cuMemFree->v2 ptr=0x%llx result=%d", (unsigned long long)pointer, result);
    return result;
}

CUresult WINAPI cuMemFreeHost(void* pointer)
{
    CUMEMFREEHOST_FN function = (CUMEMFREEHOST_FN)find_zluda_symbol("cuMemFreeHost");
    CUresult result = function != NULL ? function(pointer) : missing("cuMemFreeHost");
    TRACE_CALL(result, "cuMemFreeHost ptr=%p result=%d", pointer, result);
    return result;
}

CUresult WINAPI cuMemcpy2D(const void* copy)
{
    CUMEMCPY2D_FN function = (CUMEMCPY2D_FN)find_zluda_symbol("cuMemcpy2D_v2");
    CUresult result = function != NULL ? function(copy) : missing("cuMemcpy2D_v2");
    TRACE_CALL(result, "cuMemcpy2D->v2 descriptor=%p result=%d", copy, result);
    return result;
}

CUresult WINAPI cuMemcpyHtoDAsync(CUdeviceptr destination, const void* source, size_t bytes, CUstream stream)
{
    CUMEMCPYHTODASYNC_FN function = (CUMEMCPYHTODASYNC_FN)find_zluda_symbol("cuMemcpyHtoDAsync_v2");
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
    CUMIPMAPPEDARRAYDESTROY_FN function =
        (CUMIPMAPPEDARRAYDESTROY_FN)find_zluda_symbol("cuMipmappedArrayDestroy");
    CUresult result = function != NULL ? function(array) : missing("cuMipmappedArrayDestroy");
    tracef("cuMipmappedArrayDestroy array=%p result=%d", array, result);
    return result;
}

CUresult WINAPI cuDestroyExternalMemory(CUexternalMemory memory)
{
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
typedef uintptr_t D4rArg;
#define D4R_FORWARD(name, target, count, params, args) \
    CUresult WINAPI name params \
    { \
        typedef CUresult(__attribute__((sysv_abi)) * function_type) params; \
        function_type function = (function_type)find_zluda_symbol(#target); \
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
typedef int (*HIP_STREAM_WAIT_VALUE32_FN)(void*, void*, uint32_t, unsigned int, uint32_t);
typedef int (*HIP_STREAM_WRITE_VALUE32_FN)(void*, void*, uint32_t, unsigned int);
typedef int (*HIP_STREAM_CREATE_WITH_FLAGS_FN)(void**, unsigned int);
typedef int (*HIP_STREAM_SYNCHRONIZE_FN)(void*);

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
