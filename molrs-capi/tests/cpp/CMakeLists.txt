cmake_minimum_required(VERSION 3.14)
project(molrs_capi_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# Path to molrs workspace root (two levels up from tests/cpp/)
set(MOLRS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(MOLRS_CAPI_ROOT "${MOLRS_ROOT}")

# Cargo build profile
set(CARGO_PROFILE "debug" CACHE STRING "Cargo profile: debug or release")
if(CARGO_PROFILE STREQUAL "release")
    set(MOLRS_CARGO_PROFILE_ARGS --release)
elseif(NOT CARGO_PROFILE STREQUAL "debug")
    message(FATAL_ERROR "CARGO_PROFILE must be debug or release")
endif()
# All workspace roots in this repo share one <repo>/target via the committed
# .cargo/config.toml. Respect a local Cargo override for isolated verification.
if(DEFINED ENV{CARGO_TARGET_DIR} AND NOT "$ENV{CARGO_TARGET_DIR}" STREQUAL "")
    set(MOLRS_CARGO_TARGET_DEFAULT "$ENV{CARGO_TARGET_DIR}")
else()
    set(MOLRS_CARGO_TARGET_DEFAULT "${MOLRS_ROOT}/../target")
endif()
set(CARGO_TARGET_DIR "${MOLRS_CARGO_TARGET_DEFAULT}" CACHE PATH "Cargo target directory")
get_filename_component(CARGO_TARGET_DIR "${CARGO_TARGET_DIR}" ABSOLUTE BASE_DIR "${MOLRS_ROOT}")
set(MOLRS_TARGET_DIR "${CARGO_TARGET_DIR}/${CARGO_PROFILE}")

# --------------------------------------------------------------------------
# GoogleTest via FetchContent
# --------------------------------------------------------------------------
include(FetchContent)
FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG        v1.15.2
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

# --------------------------------------------------------------------------
# Build Rust library (custom target)
# --------------------------------------------------------------------------
# The cdylib, not the staticlib: this test must exercise the same artifact a C
# consumer loads. Suffix/prefix come from CMake because ci-capi runs ubuntu
# (.so), not macOS (.dylib).
set(MOLRS_SHARED_LIB
    "${MOLRS_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}molrs_capi${CMAKE_SHARED_LIBRARY_SUFFIX}")

# Let Cargo check source fingerprints on every build; an existing library
# alone does not mean the Rust sources or generated C header are up to date.
add_custom_target(molrs_capi_build
    COMMAND cargo build --manifest-path "${MOLRS_CAPI_ROOT}/Cargo.toml"
            --target-dir "${CARGO_TARGET_DIR}" ${MOLRS_CARGO_PROFILE_ARGS}
    BYPRODUCTS "${MOLRS_SHARED_LIB}"
    WORKING_DIRECTORY "${MOLRS_ROOT}"
    COMMENT "Building molrs-capi (Rust shared library)"
    VERBATIM
)

# --------------------------------------------------------------------------
# Imported library target
# --------------------------------------------------------------------------
add_library(molrs_capi SHARED IMPORTED)
set_target_properties(molrs_capi PROPERTIES
    IMPORTED_LOCATION ${MOLRS_SHARED_LIB}
)

# No system-library interface list here: that was the staticlib era, where the
# consumer had to re-declare everything Rust's runtime pulls in. A cdylib
# records its own dependencies.

target_include_directories(molrs_capi INTERFACE
    ${MOLRS_CAPI_ROOT}/include
)

# --------------------------------------------------------------------------
# Test executable
# --------------------------------------------------------------------------
add_executable(test_molrs_capi test_molrs_capi.cpp)
add_dependencies(test_molrs_capi molrs_capi_build)
target_link_libraries(test_molrs_capi
    molrs_capi
    GTest::gtest_main
)

# The only rpath this binary needs: libmolrs_capi's own directory. On ELF its
# SONAME is the bare `libmolrs_capi.so`, so the loader has to be told where to
# look. Everything below it is already self-locating: libmolrs_capi carries
# molrs inside itself.
set_target_properties(test_molrs_capi PROPERTIES
    BUILD_RPATH "${MOLRS_TARGET_DIR}"
    INSTALL_RPATH "${MOLRS_TARGET_DIR}"
)

enable_testing()
add_test(NAME molrs_capi_tests COMMAND test_molrs_capi)
