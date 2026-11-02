// Same 304-byte parameter ABI through either a typed SRV or a root CBV.
// uint4 keeps array elements tightly packed in constant-buffer layout.
#ifndef D4R_M_LAYER_CBV
#define D4R_M_LAYER_CBV 0
#endif
#if D4R_M_LAYER_CBV
cbuffer LayerParameters : register(b0) { uint4 layerWords[19]; }
uint mParameter(uint i) { return layerWords[i >> 2][i & 3]; }
#else
Buffer<uint> parameters : register(t0);
uint mParameter(uint i) { return parameters[i]; }
#endif
uint off(uint i) { return mParameter(4 + i); }
