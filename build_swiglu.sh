#!/bin/bash

XNNPACK_BUILD_DIR="XNNPACK/build/local"

g++ minimal_swiglu.cpp -o minimal_swiglu_kernel \
    -I XNNPACK/include \
    -I ${XNNPACK_BUILD_DIR}/include \
    -I ${XNNPACK_BUILD_DIR}/pthreadpool-source/include \
    -L ${XNNPACK_BUILD_DIR} \
    -L ${XNNPACK_BUILD_DIR}/pthreadpool \
    -L ${XNNPACK_BUILD_DIR}/cpuinfo \
    -lXNNPACK \
    -lxnnpack-microkernels-prod \
    -lpthreadpool \
    -lcpuinfo \
    -lm \
    -lpthread