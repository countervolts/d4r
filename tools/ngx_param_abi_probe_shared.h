#pragma once

struct D4rNgxParamLogRecord
{
    unsigned char operation;
    unsigned char valueType;
    unsigned short reserved;
    unsigned int result;
    unsigned long long value;
    char name[64];
};
static_assert(sizeof(D4rNgxParamLogRecord) == 80);
