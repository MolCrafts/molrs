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
# All workspace roots in this repo share one <repo>/target via the committed
# .cargo/config.toml, so the cdylib lands one level above the crate dir.
set(MOLRS_TARGET_DIR "${MOLRS_ROOT}/../target/${CARGO_PROFILE}")

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
# The cdylib, not the staticlib: locally every native consumer resolves molrs
# through the one shared libmolrs_ffi dylib, and a C++ test that statically
# embedded its own molrs would be testing a second image (see
# .claude/specs/ffi-shared-dylib.md). Suffix/prefix come from CMake because
# ci-capi runs ubuntu (.so), not macOS (.dylib).
set(MOLRS_SHARED_LIB
    "${MOLRS_TARGET_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}molrs_capi${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_custom_command(
    OUTPUT ${MOLRS_SHARED_LIB}
    COMMAND cargo build --manifest-path ${MOLRS_CAPI_ROOT}/Cargo.toml
    WORKING_DIRECTORY ${MOLRS_ROOT}
    COMMENT "Building molrs-capi (Rust shared library)"
)
add_custom_target(molrs_capi_build DEPENDS ${MOLRS_SHARED_LIB})

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
# look. Everything below it is already self-locating — libmolrs_capi resolves
# libmolrs_ffi by the hashless absolute install_name of the shared target dir,
# and carries the libstd rpath injected by molrs-capi/build.rs.
set_target_properties(test_molrs_capi PROPERTIES
    BUILD_RPATH "${MOLRS_TARGET_DIR}"
    INSTALL_RPATH "${MOLRS_TARGET_DIR}"
)

enable_testing()
add_test(NAME molrs_capi_tests COMMAND test_molrs_capi)
