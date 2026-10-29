# The LSFIRE fire engine directory, for CMake. ERF includes this file when
# ERF_FIRE_ENGINE_DIRS lists this directory and calls erf_fire_engine_add with
# its object library (ERF_FIRE_ENGINE_DIR is this directory during the call).
#
# The LSFIRE core (LSFIRE/include, LSFIRE/src) is compiled into ERF's library
# with ERF's own AMReX; there is no separate libelmfire_core.a.
set(LSFIRE_CORE_SOURCES
  LSFIRE/src/Core/ElmfireEngine.cpp
  LSFIRE/src/Core/ConfigReader.cpp
  LSFIRE/src/Core/OutputWriter.cpp
  LSFIRE/src/Core/ElmfireAmrCore.cpp
  LSFIRE/src/Core/FireHierarchy.cpp
  LSFIRE/src/Physics/FuelModels.cpp
  LSFIRE/src/Physics/Rothermel.cpp
  LSFIRE/src/Physics/EllipticalFire.cpp
  LSFIRE/src/Physics/Spotting.cpp
  LSFIRE/src/Numerics/Gradients.cpp
  LSFIRE/src/Numerics/LevelSetSolver.cpp
  LSFIRE/src/DataStructures/GpuNarrowBand.cpp
  LSFIRE/src/API/ELMFIRE_API.cpp)

function(erf_fire_engine_add target)
  set(_sources
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireELMFIRE.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireReadFuel.cpp
    ${ERF_FIRE_ENGINE_DIR}/ERF_FireReadIgnition.cpp)
  foreach(_src IN LISTS LSFIRE_CORE_SOURCES)
    list(APPEND _sources ${ERF_FIRE_ENGINE_DIR}/${_src})
  endforeach()
  target_sources(${target} PRIVATE ${_sources})
  if(ERF_ENABLE_CUDA)
    # Not yet built or tested on GPU
    set_source_files_properties(${_sources} PROPERTIES LANGUAGE CUDA)
  endif()
  target_include_directories(${target} PUBLIC
    $<BUILD_INTERFACE:${ERF_FIRE_ENGINE_DIR}>
    $<BUILD_INTERFACE:${ERF_FIRE_ENGINE_DIR}/LSFIRE/include>
    $<BUILD_INTERFACE:${ERF_FIRE_ENGINE_DIR}/LSFIRE/src>)   # Core/FireHierarchy.H
  message(STATUS "LSFIRE fire engine: core compiled from ${ERF_FIRE_ENGINE_DIR}/LSFIRE")
endfunction()
