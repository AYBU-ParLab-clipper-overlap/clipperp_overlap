cmake_minimum_required(VERSION 3.10)
project(clipperplus VERSION 1.0.0 LANGUAGES C CXX)


# ------------------------------------------------------------------
# Default build type: RelWithDebInfo
# ------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "No build type selected, defaulting to RelWithDebInfo")
    set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING
        "Choose the type of build: Debug Release RelWithDebInfo MinSizeRel"
        FORCE
    )
endif()

###############################################################################
# Build settings
###############################################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Default build type → Debug (DO NOT force Release)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug CACHE STRING "Build type" FORCE)
endif()

###############################################################################
# Options
###############################################################################

option(DEBUG_FLAG        "Enable debugging output" ON)
option(DEBUG_TIMING_FLAG "Enable timing report" ON)
option(DEBUG_OPTIM_FLAG  "Enable debugging clipper optimization" ON)

option(BUILD_PMC_HEU             "Build PMC heuristic" ON)
option(BUILD_BINDINGS_MATLAB     "Build MATLAB bindings" OFF)
option(BUILD_BINDINGS_PYTHON     "Build Python bindings" OFF)
option(BUILD_TESTS               "Build testsuite" OFF)

###############################################################################
# Dependencies
###############################################################################

find_package(Eigen3 REQUIRED)
message(STATUS "Eigen version: ${EIGEN3_VERSION_STRING}")

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

###############################################################################
# METIS
###############################################################################

set(METIS_DIR "/apps/GPP/EASYBUILD/software/METIS/5.1.0-GCCcore-13.2.0")

find_library(METIS_LIB metis
    PATHS ${METIS_DIR}/lib
    REQUIRED
)

find_path(METIS_INCLUDE_DIR metis.h
    PATHS ${METIS_DIR}/include
    REQUIRED
)

message(STATUS "METIS library: ${METIS_LIB}")

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

target_include_directories(clipperplus
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${METIS_INCLUDE_DIR}
)

target_link_libraries(clipperplus
    PUBLIC
        Eigen3::Eigen
        OpenMP::OpenMP_CXX
        MPI::MPI_CXX
        ${METIS_LIB}
)

# ---- DEBUG MACROS (MATCH OPTION NAMES EXACTLY) ----
if(DEBUG_FLAG)
    target_compile_definitions(clipperplus PRIVATE DEBUG_FLAG)
endif()

set_target_properties(clipperplus PROPERTIES
    VERSION ${PROJECT_VERSION}
)

target_compile_definitions(clipperplus
    PUBLIC CLIPPERPLUS_VERSION="${PROJECT_VERSION}"
)

###############################################################################
# Executable: demo_clipper
###############################################################################

add_executable(demo_clipper
    src/demo_clipper.cpp
)

if(DEBUG_TIMING_FLAG)
    target_compile_definitions(clipperplus PRIVATE DEBUG_TIMING)
    target_compile_definitions(demo_clipper PRIVATE DEBUG_TIMING)
endif()

if(DEBUG_OPTIM_FLAG)
    target_compile_definitions(clipperplus PRIVATE DEBUG_OPTIM)
    target_compile_definitions(demo_clipper PRIVATE DEBUG_OPTIM)
endif()


target_include_directories(demo_clipper
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(demo_clipper
    PRIVATE clipperplus
)

# ---- SAME DEBUG MACROS FOR EXECUTABLE ----
if(DEBUG_FLAG)
    target_compile_definitions(demo_clipper PRIVATE DEBUG_FLAG)
endif()

if(DEBUG_OPTIM_FLAG)
    target_compile_definitions(demo_clipper PRIVATE DEBUG_OPTIM_FLAG)
endif()

if(DEBUG_TIMING_FLAG)
    target_compile_definitions(demo_clipper PRIVATE DEBUG_TIMING_FLAG)
endif()

###############################################################################
# Optional components
###############################################################################

if(BUILD_BINDINGS_MATLAB)
    message(STATUS "Building MATLAB bindings")
    add_subdirectory(bindings/matlab)
endif()

if(BUILD_BINDINGS_PYTHON)
    message(STATUS "Building Python bindings")
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

