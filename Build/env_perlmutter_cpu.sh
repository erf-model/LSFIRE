#!/bin/bash
# Module environment for building/running ERF + LSFIRE (CPU) on Perlmutter.
# Usage: source Build/env_perlmutter_cpu.sh
#
# Why cpe/24.07: the system default (cpe/26.03) pairs gcc 14 + cray-mpich 9.1.0 +
# cray-libsci 26.03. Mixing gcc 12.3/mpich 8.1.30 with libsci 26.03 links two MPI
# libraries into one binary (libmpi_gnu_123 from 8.1.30 and libmpi_gnu.so.12 from 9.1.0).
# cpe/24.07 gives a self-consistent set: gcc 13.2, mpich 8.1.30, libsci 24.07.0,
# craype 2.7.32 (same versions as the old ERF-ELMFIRE mybuild_cpu/cmake.sh).

module unload gpu cudatoolkit craype-accel-nvidia80 cray-libsci_acc 2>/dev/null
module load cpu
module load cpe/24.07
module load craype/2.7.32
module load gcc-native/13.2
module load cray-mpich/8.1.30
module load cray-libsci/24.07.0
module load cray-hdf5-parallel/1.14.3.1
module load cray-netcdf-hdf5parallel/4.9.0.13
module load cray-parallel-netcdf/1.12.3.13
module load cray-fftw/3.3.10.8
module load cmake/3.30.2 2>/dev/null || true

# Non-default PE versions: make the runtime loader pick them over /opt/cray/pe/lib64 defaults
export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
