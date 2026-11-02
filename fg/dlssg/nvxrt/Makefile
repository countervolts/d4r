CXX ?= g++
CXXFLAGS ?= -O2 -g -std=c++17 -fPIC -fvisibility=hidden -Wall -Wno-unused-function
LDFLAGS ?= -shared -static-libstdc++ -static-libgcc -Wl,--no-undefined -Wl,-z,defs -pthread
OUT = build
all: $(OUT)/libVkLayer_nvxrt.so $(OUT)/nvxrt_layer.json $(OUT)/nvxrt-scan
$(OUT)/libVkLayer_nvxrt.so: nvxrt.cpp util.cpp util.h
	mkdir -p $(OUT)
	$(CXX) $(CXXFLAGS) -o $@ nvxrt.cpp util.cpp $(LDFLAGS)
$(OUT)/nvxrt_layer.json: nvxrt_layer.json
	mkdir -p $(OUT)
	cp $< $@
clean:
	rm -rf $(OUT)
$(OUT)/nvxrt-scan: scan.cpp util.cpp util.h
	mkdir -p $(OUT)
	$(CXX) -O2 -std=c++17 -Wall -o $@ scan.cpp util.cpp
