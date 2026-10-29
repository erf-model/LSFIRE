#!/bin/bash
# Configure and build erf_exec with the LSFIRE fire engine (CPU, MPI).
# Usage: bash Build/cmake_cpu.sh <build dir> [jobs]
# On Perlmutter, source Build/env_perlmutter_cpu.sh first.
set -e
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
BUILD=${1:?usage: cmake_cpu.sh <build dir> [jobs]}
JOBS=${2:-16}

cmake -S ${REPO}/ERF -B ${BUILD} \
  -DCMAKE_BUILD_TYPE=Release \
  -DERF_DIM=3 \
  -DERF_ENABLE_MPI=ON \
  -DERF_ENABLE_CUDA=OFF \
  -DERF_ENABLE_NETCDF=ON \
  -DERF_ENABLE_TESTS=OFF \
  -DERF_ENABLE_FIRE=ON \
  -DERF_FIRE_ENGINE_DIRS=${REPO}/FireEngine \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
cmake --build ${BUILD} --target erf_exec -j ${JOBS}
