/* PCI identity comparisons shared by the Wine bridge and CPU tests. */
#pragma once
#include <stdio.h>

typedef struct D4rPciAddress { unsigned domain, bus, device, function; } D4rPciAddress;

static int d4r_parse_pci(const char* text, D4rPciAddress* out)
{
    char extra;
    return text != NULL && sscanf(text, "%x:%x:%x.%x%c", &out->domain, &out->bus,
        &out->device, &out->function, &extra) == 4 && out->bus <= 255 && out->device <= 31 && out->function <= 7;
}

static int d4r_same_pci(const char* a, const char* b)
{
    D4rPciAddress x, y;
    return d4r_parse_pci(a, &x) && d4r_parse_pci(b, &y) && x.domain == y.domain &&
        x.bus == y.bus && x.device == y.device && x.function == y.function;
}

static int d4r_pci_matches_kfd(const char* pci, unsigned long domain, unsigned long location)
{
    D4rPciAddress address;
    return d4r_parse_pci(pci, &address) && address.domain == domain &&
        ((address.bus << 8) | (address.device << 3) | address.function) == location;
}
