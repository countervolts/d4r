/* Native directory selection shared with the CPU-only configuration tests. */
#pragma once
#include <dirent.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int d4r_apply_accuracy_policy(void)
{
    const char* prefer = getenv("D4R_PREFER_ACCURACY");
    if (prefer == NULL || strcmp(prefer, "1") != 0)
        return 0;
    setenv("D4R_ZLUDA_IGNORE_DENORMAL", "0", 1);
    setenv("D4R_ZLUDA_WMMA_F32ACC", "0", 1);
    setenv("D4R_ZLUDA_FAST_MATH", "0", 1);
    setenv("D4R_ZLUDA_WAVE64", "0", 1);
    setenv("D4R_ELIDE_NGX_SYNC", "0", 1);
    return 1;
}

static int d4r_accuracy_directory(const char* directory)
{
    char path[1200], version[16];
    if (snprintf(path, sizeof(path), "%s/d4r-accuracy.txt", directory) >= (int)sizeof(path))
        return 0;
    FILE* file = fopen(path, "r");
    if (file == NULL)
        return 0;
    const int valid = fgets(version, sizeof(version), file) != NULL && strcmp(version, "1\n") == 0;
    fclose(file);
    return valid;
}

/* 1 = hash-checked set, 2 = unmanifested developer set, 0 = no usable set.
   Accuracy mode never selects an unmarked directory, including flat legacy developer builds. */
static int d4r_native_candidate(const char* directory, int accuracy)
{
    if (accuracy && !d4r_accuracy_directory(directory))
        return 0;
    char path[1200];
    if (snprintf(path, sizeof(path), "%s/d4r-kernels.txt", directory) >= (int)sizeof(path))
        return 0;
    FILE* file = fopen(path, "r");
    if (file != NULL)
    {
        fclose(file);
        return 1;
    }
    return accuracy ? 2 : 0;
}

static int d4r_select_native_source(const char* configured, const char* architecture, int accuracy,
                                  int fp8, char* source, size_t size)
{
    char base[1024];
    const int length = accuracy && !d4r_accuracy_directory(configured)
        ? snprintf(base, sizeof(base), "%s/accuracy", configured)
        : snprintf(base, sizeof(base), "%s", configured);
    if (length < 0 || length >= (int)sizeof(base))
        return 0;
    int kind;
    if (fp8 && strncmp(architecture, "gfx12", 5) == 0)
    {
        if (snprintf(source, size, "%s/%s-fp8", base, architecture) >= (int)size)
            return 0;
        if ((kind = d4r_native_candidate(source, accuracy)) != 0)
            return kind;
    }
    if (architecture[0] != '\0')
    {
        if (snprintf(source, size, "%s/%s", base, architecture) >= (int)size)
            return 0;
        if ((kind = d4r_native_candidate(source, accuracy)) != 0)
            return kind;
    }
    if (snprintf(source, size, "%s", base) >= (int)size)
        return 0;
    if ((kind = d4r_native_candidate(source, accuracy)) != 0)
        return kind;
    if (accuracy)
        return 0;
    /* Keep legacy flat developer directories working, but do not hand a release root to ZLUDA. */
    DIR* directory = opendir(base);
    if (directory == NULL)
        return 0;
    int targets = 0;
    for (struct dirent* entry; (entry = readdir(directory)) != NULL;)
        targets += strncmp(entry->d_name, "gfx", 3) == 0;
    closedir(directory);
    return targets ? 0 : 2;
}
