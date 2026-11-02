#!/usr/bin/env python3
"""Where the CNN Ray Reconstruction model's tensors are in nvngx_dlssd.dll 310.3.0 (presets A-C).

The DLL keeps every tensor NGX uploads for the CNN layers ("Enc0" ... "PDL") as plain binary16 at its full shape
(zero rows included), three times; NGX uploads them unchanged. The offsets below are the first copy of each, found
by matching harness captures of presets A and B (C runs B's model) byte for byte, 2026-10-17. They hold for the one
DLL file whose SHA-256 is DLL_SHA256 (NVIDIA's DLSS SDK 310.3.0, github.com/NVIDIA/DLSS a8ed84e); any other file is
refused. Nothing of the DLL is in this file: the weights are read from the user's copy at build time.
"""

DLL_SHA256 = 'e43fdcead51c388390fd74211dcc4ee8c218e7650d4f7ce3e8ca9d899db7f33a'

# stage -> (input channels, output channels)
CHANNELS = {'Enc0': (16, 32), 'Enc1': (32, 64), 'Enc2': (64, 96), 'Enc3': (96, 128), 'Enc4': (128, 160),
            'Dec4': (160, 160), 'Dec3': (160, 128), 'Dec2': (128, 96), 'Dec1': (96, 64), 'Dec0': (64, 32), 'PDL': (32, 32)}

# preset model -> "Stage.tensor" -> (byte offset in the DLL, bytes)
TENSORS = {
    'A': {
        'Enc0.w1': (0x149760, 9216), 'Enc0.w2': (0x150760, 2048), 'Enc0.w3': (0x150f60, 1024),
        'Enc0.b1': (0x150720, 64), 'Enc0.b2': (0x151b60, 64), 'Enc0.b3': (0x149740, 32),
        'Enc1.w1': (0x165c10, 36864), 'Enc1.w2': (0x1abf40, 8192), 'Enc1.w3': (0x1af3e0, 2048),
        'Enc1.b1': (0x175d80, 128), 'Enc1.b2': (0x1adf40, 128), 'Enc1.b3': (0x1afbe0, 32),
        'Enc2.w1': (0x1b7110, 110592), 'Enc2.w2': (0x1d21d0, 18432), 'Enc2.w3': (0x1d6a90, 3072),
        'Enc2.b1': (0x151e00, 192), 'Enc2.b2': (0x1d69d0, 192), 'Enc2.b3': (0x175d60, 32),
        'Enc3.w1': (0x1df710, 221184), 'Enc3.w2': (0x1d7690, 32768), 'Enc3.w3': (0x222010, 4096),
        'Enc3.b1': (0x151ba0, 256), 'Enc3.b2': (0x221f10, 256), 'Enc3.b3': (0x223010, 32),
        'Enc4.w1': (0x22a390, 368640), 'Enc4.w2': (0x2844d0, 51200), 'Enc4.w3': (0x1adfc0, 5120),
        'Enc4.b1': (0x284390, 320), 'Enc4.b2': (0x175e00, 320), 'Enc4.b3': (0x1afc00, 32),
        'Dec4.w1': (0x2988d0, 460800), 'Dec4.w2': (0x159410, 51200), 'Dec4.w3': (0x309230, 51200),
        'Dec4.w4': (0x316770, 5120), 'Dec4.b1': (0x151cc0, 320), 'Dec4.b2': (0x3090f0, 320),
        'Dec4.b3': (0x316630, 320), 'Dec4.b4': (0x149720, 32), 'Dec3.w1': (0x321c50, 368640),
        'Dec3.w2': (0x219f10, 32768), 'Dec3.w3': (0x380550, 32768), 'Dec3.w4': (0x388650, 4096),
        'Dec3.b1': (0x149620, 256), 'Dec3.b2': (0x380450, 256), 'Dec3.b3': (0x388550, 256),
        'Dec3.b4': (0x389650, 32), 'Dec2.w1': (0x175f40, 221184), 'Dec2.w2': (0x215710, 18432),
        'Dec2.w3': (0x37bc50, 18432), 'Dec2.w4': (0x315a30, 3072), 'Dec2.b1': (0x392bd0, 192),
        'Dec2.b2': (0x392c90, 192), 'Dec2.b3': (0x1d2110, 192), 'Dec2.b4': (0x151ca0, 32),
        'Dec1.w1': (0x392d50, 110592), 'Dec1.w2': (0x3be400, 8192), 'Dec1.w3': (0x3c0480, 8192),
        'Dec1.w4': (0x3c2500, 2048), 'Dec1.b1': (0x1df690, 128), 'Dec1.b2': (0x3c0400, 128),
        'Dec1.b3': (0x3c2480, 128), 'Dec1.b4': (0x3090d0, 32), 'Dec0.w1': (0x3ccf60, 36864),
        'Dec0.w2': (0x151360, 2048), 'Dec0.w3': (0x3d5fe0, 2048), 'Dec0.w4': (0x3d6820, 1024),
        'Dec0.b1': (0x3d5f60, 64), 'Dec0.b2': (0x3d5fa0, 64), 'Dec0.b3': (0x3d67e0, 64),
        'Dec0.b4': (0x1af3c0, 32), 'PDL.w1': (0x3de2c0, 18432), 'PDL.w2': (0x321450, 2048),
        'PDL.w3': (0x3e2b40, 2048), 'PDL.b1': (0x3e2ac0, 64), 'PDL.b2': (0x3e2b00, 64),
        'PDL.b3': (0x3e3340, 64),
    },
    'B': {
        'Enc0.w1': (0xde5150, 9216), 'Enc0.w2': (0xd32570, 2048), 'Enc0.w3': (0xe2dd90, 1024),
        'Enc0.b1': (0xe2dd50, 64), 'Enc0.b2': (0xd2a530, 64), 'Enc0.b3': (0xde8250, 32),
        'Enc1.w1': (0xe9f5d0, 36864), 'Enc1.w2': (0xeed430, 8192), 'Enc1.w3': (0xeef6f0, 2048),
        'Enc1.b1': (0xea85f0, 128), 'Enc1.b2': (0xeef530, 128), 'Enc1.b3': (0xe2e1b0, 32),
        'Enc2.w1': (0xefc7f0, 110592), 'Enc2.w2': (0xf719b0, 18432), 'Enc2.w3': (0xe9e9d0, 3072),
        'Enc2.b1': (0xea8670, 192), 'Enc2.b2': (0xf761b0, 192), 'Enc2.b3': (0xf76270, 32),
        'Enc3.w1': (0xf76290, 221184), 'Enc3.w2': (0xfac3d0, 32768), 'Enc3.w3': (0xfb43d0, 4096),
        'Enc3.b1': (0xeef430, 256), 'Enc3.b2': (0xf718b0, 256), 'Enc3.b3': (0xea85d0, 32),
        'Enc4.w1': (0xf178b0, 368640), 'Enc4.w2': (0xfb53d0, 51200), 'Enc4.w3': (0xfc1d30, 5120),
        'Enc4.b1': (0xfac290, 320), 'Enc4.b2': (0xfc1bf0, 320), 'Enc4.b3': (0xfc7930, 32),
        'Dec4.w1': (0xe2e1d0, 460800), 'Dec4.w2': (0x1012070, 51200), 'Dec4.w3': (0xeefff0, 51200),
        'Dec4.w4': (0x101e9b0, 5120), 'Dec4.b1': (0x1011f30, 320), 'Dec4.b2': (0xeef5b0, 320),
        'Dec4.b3': (0x101e870, 320), 'Dec4.b4': (0xe2e190, 32), 'Dec3.w1': (0x107e010, 368640),
        'Dec3.w2': (0xd2a570, 32768), 'Dec3.w3': (0x10d8110, 32768), 'Dec3.w4': (0x10e0110, 4096),
        'Dec3.b1': (0xde8150, 256), 'Dec3.b2': (0x10d8010, 256), 'Dec3.b3': (0xeefef0, 256),
        'Dec3.b4': (0xfc1bd0, 32), 'Dec2.w1': (0x113b5f0, 221184), 'Dec2.w2': (0x11715f0, 18432),
        'Dec2.w3': (0xfc3130, 18432), 'Dec2.w4': (0xde7550, 3072), 'Dec2.b1': (0xf177f0, 192),
        'Dec2.b2': (0xd5e5d0, 192), 'Dec2.b3': (0x1175df0, 192), 'Dec2.b4': (0x1175eb0, 32),
        'Dec1.w1': (0x1204fb0, 110592), 'Dec1.w2': (0x1220030, 8192), 'Dec1.w3': (0x12228b0, 8192),
        'Dec1.w4': (0x1224930, 2048), 'Dec1.b1': (0x121ffb0, 128), 'Dec1.b2': (0x1222030, 128),
        'Dec1.b3': (0x12248b0, 128), 'Dec1.b4': (0x1225130, 32), 'Dec0.w1': (0x127aab0, 36864),
        'Dec0.w2': (0x1283af0, 2048), 'Dec0.w3': (0x1284330, 2048), 'Dec0.w4': (0x1284b70, 1024),
        'Dec0.b1': (0x1283ab0, 64), 'Dec0.b2': (0x12842f0, 64), 'Dec0.b3': (0x1284b30, 64),
        'Dec0.b4': (0xea8730, 32), 'PDL.w1': (0x12c0fd0, 18432), 'PDL.w2': (0x12c5810, 2048),
        'PDL.w3': (0x12220b0, 2048), 'PDL.b1': (0x12c57d0, 64), 'PDL.b2': (0x12c6010, 64),
        'PDL.b3': (0x12c6050, 64),
    },
}
