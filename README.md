# LSFIRE fire engine for ERF

LSFIRE is a level-set wildland fire spread model (Rothermel spread, ELMFIRE-style
narrow band) built on AMReX. This repository couples it to the
[ERF](https://github.com/erf-model/ERF) atmospheric model as a fire engine behind
ERF's shared fire coupling (erf-model/ERF#4798): `erf.fire.engine = lsfire`.

## Layout

| Path | Contents |
|---|---|
| `ERF/` | ERF, as a git submodule pinned to an upstream commit; not modified |
| `FireEngine/` | The engine directory ERF builds in (`ERFFireEngine.cmake`, `Make.package`) and the ERF adapter (`ERF_FireELMFIRE.*`, fuel and ignition readers) |
| `FireEngine/LSFIRE/` | The LSFIRE core (`include/`, `src/`), compiled into `erf_exec` with ERF's AMReX |
| `Exec/RegTests/LSFIRE/` | Small coupled test cases and their reference results |
| `Build/` | Build scripts (Perlmutter CPU environment, CMake) |

## Build

```
git clone --recursive <this repository>     # or: git submodule update --init ERF && \
                                            #     git -C ERF submodule update --init Submodules/AMReX
source Build/env_perlmutter_cpu.sh          # Perlmutter only
bash Build/cmake_cpu.sh <build dir>
```

The executable is `<build dir>/Exec/erf_exec`. See `Exec/RegTests/LSFIRE/README.md` for the
inputs and the GNU make build.

## Status

- The fire runs on one fire level coupled to ERF level 0. LSFIRE's static multilevel
  fire grid is in the source but is not supported through this engine.
- CPU only so far; GPU builds are not tested.
- Fire restart (checkpointing) is not implemented; a restart with the fire on aborts.

## Sources

| Component | Origin |
|---|---|
| `FireEngine/LSFIRE/` | liuye923/LSFIRE, branch `multilevel_static`, commit 86e84af (`include/`, `src/` without the unused `NarrowBandTracker`) |
| `FireEngine/` adapter, `Exec/RegTests/LSFIRE/` | liuye923/ERF-LSFIRE PR #6 (commit 9d71a51be), build files changed to compile the LSFIRE core from `FireEngine/LSFIRE` |
| `ERF/` | erf-model/ERF `development`, commit 1d14da148 |
