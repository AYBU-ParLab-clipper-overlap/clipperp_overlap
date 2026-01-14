cmake_minimum_required(VERSION 3.10)
project(clipperplus_overlap VERSION 1.0.0)

###############################################################################
# Build settings
###############################################################################
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type: RelWithDebInfo (single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
      "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel" FORCE)
endif()

###############################################################################
# Options
###############################################################################
option(DEBUG_FLAG        "Enable debugging output" ON)
option(DEBUG_TIMING_FLAG "Enable timing report"     ON)
option(DEBUG_OPTIM_FLAG  "Enable optimization debug" ON)

option(BUILD_BINDINGS_MATLAB "Build MATLAB bindings" OFF)
option(BUILD_BINDINGS_PYTHON "Build Python bindings" OFF)
option(BUILD_TESTS           "Build testsuite"       OFF)

###############################################################################
# Dependencies
###############################################################################
find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

message(STATUS "Eigen version: ${EIGEN3_VERSION_STRING}")

###############################################################################
# METIS & GKlib
###############################################################################
set(METIS_DIR "/arf/home/esalman/libraries/metis-install/")
set(GKLIB_DIR "/arf/home/esalman/libraries/gklib-install")

find_library(METIS_LIB metis PATHS ${METIS_DIR}/lib REQUIRED)
find_path(METIS_INCLUDE_DIR metis.h PATHS ${METIS_DIR}/include REQUIRED)

find_library(GKLIB_LIB GKlib PATHS ${GKLIB_DIR}/lib REQUIRED)
find_path(GKLIB_INCLUDE_DIR gk_types.h PATHS ${GKLIB_DIR}/include REQUIRED)

message(STATUS "METIS library: ${METIS_LIB}")
message(STATUS "GKlib library: ${GKLIB_LIB}")

###############################################################################
# Sources
###############################################################################
set(CLIPPERPLUS_SRC
  src/clipperplus_clique.cpp
  src/clipperplus_graph.cpp
  src/clique_optimization.cpp
  src/clipperplus_heuristic.cpp
  src/utils.cpp
)

###############################################################################
# Library: clipperplus
###############################################################################
add_library(clipperplus SHARED ${CLIPPERPLUS_SRC})

target_include_directories(clipperplus PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${METIS_INCLUDE_DIR}
  ${GKLIB_INCLUDE_DIR}
)

target_link_libraries(clipperplus PUBLIC
  Eigen3::Eigen
  OpenMP::OpenMP_CXX
  MPI::MPI_CXX
  ${METIS_LIB}
  ${GKLIB_LIB}
)

target_compile_definitions(clipperplus PUBLIC
  CLIPPERPLUS_VERSION="${PROJECT_VERSION}"
)

# Debug macros (match your C++ #ifdef names)
if(DEBUG_FLAG)
  target_compile_definitions(clipperplus PRIVATE DEBUG_FLAG)
endif()
if(DEBUG_TIMING_FLAG)
  target_compile_definitions(clipperplus PRIVATE DEBUG_TIMING)
endif()
if(DEBUG_OPTIM_FLAG)
  target_compile_definitions(clipperplus PRIVATE DEBUG_OPTIM)
endif()

set_target_properties(clipperplus PROPERTIES VERSION ${PROJECT_VERSION})

###############################################################################
# Executable: demo_clipper
###############################################################################
add_executable(demo_clipper src/demo_clipper.cpp)

target_include_directories(demo_clipper PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(demo_clipper PRIVATE clipperplus)

# Same debug macros for the executable
if(DEBUG_FLAG)
  target_compile_definitions(demo_clipper PRIVATE DEBUG_FLAG)
endif()
if(DEBUG_TIMING_FLAG)
  target_compile_definitions(demo_clipper PRIVATE DEBUG_TIMING)
endif()
if(DEBUG_OPTIM_FLAG)
  target_compile_definitions(demo_clipper PRIVATE DEBUG_OPTIM)
endif()

###############################################################################
# Optional components
###############################################################################
if(BUILD_BINDINGS_MATLAB)
  add_subdirectory(bindings/matlab)
endif()

if(BUILD_BINDINGS_PYTHON)
  add_subdirectory(bindings/python)
endif()

if(BUILD_TESTS)
  enable_testing()
  add_subdirectory(test)
endif()

###############################################################################
# Summary
###############################################################################
message(STATUS "-----------------------------------------")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "DEBUG_FLAG=${DEBUG_FLAG}")
message(STATUS "DEBUG_OPTIM_FLAG=${DEBUG_OPTIM_FLAG}")
message(STATUS "DEBUG_TIMING_FLAG=${DEBUG_TIMING_FLAG}")
message(STATUS "-----------------------------------------")

