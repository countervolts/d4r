#!/bin/bash
set -e
cd "$(dirname "$0")"
mkdir -p ../../build/fg
g++ -O2 -g -std=c++17 -Wall -Wno-unused-variable -o ../../build/fg/fg_replay_test fg_replay_test.cpp fg_engine.cpp -ldl
