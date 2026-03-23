cmake_minimum_required(VERSION 3.14)
project(molrs_capi_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --------------------------------------------------------------------------
# Options
# --------------------------------------------------------------------------
option(MOLRS_F64 "Use double precision (must match cargo build --features f64)" OFF)

# Path to molrs workspace root (two levels up from tests/cpp/)
set(MOLRS_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../..")
set(MOLRS_CAPI_ROOT "${MOLRS_ROOT}")

# Cargo build profile
set(CARGO_PROFILE "debug" CACHE STRING "Cargo profile: debug or release")
set(MOLRS_TARGET_DIR "${MOLRS_ROOT}/target/${CARGO_PROFILE}")

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
set(MOLRS_STATIC_LIB "${MOLRS_TARGET_DIR}/libmolrs_capi.a")

if(MOLRS_F64)
    set(CARGO_FEATURES "--features" "f64")
else()
    set(CARGO_FEATURES "")
endif()

add_custom_command(
    OUTPUT ${MOLRS_STATIC_LIB}
    COMMAND cargo build --manifest-path ${MOLRS_CAPI_ROOT}/Cargo.toml ${CARGO_FEATURES}
    WORKING_DIRECTORY ${MOLRS_ROOT}
    COMMENT "Building molrs-capi (Rust static library)"
)
add_custom_target(molrs_capi_build DEPENDS ${MOLRS_STATIC_LIB})

# --------------------------------------------------------------------------
# Imported library target
# --------------------------------------------------------------------------
add_library(molrs_capi STATIC IMPORTED)
set_target_properties(molrs_capi PROPERTIES
    IMPORTED_LOCATION ${MOLRS_STATIC_LIB}
)

# System libraries required by Rust staticlib on this platform
if(APPLE)
    target_link_libraries(molrs_capi INTERFACE
        "-lc" "-lm" "-liconv" "-lSystem"
        "-framework CoreFoundation"
        "-framework Security"
    )
elseif(UNIX)
    target_link_libraries(molrs_capi INTERFACE
        "-lc" "-lm" "-ldl" "-lpthread"
    )
endif()

target_include_directories(molrs_capi INTERFACE
    ${MOLRS_CAPI_ROOT}/include
)

if(MOLRS_F64)
    target_compile_definitions(molrs_capi INTERFACE MOLRS_F64)
endif()

# --------------------------------------------------------------------------
# Test executable
# --------------------------------------------------------------------------
add_executable(test_molrs_capi test_molrs_capi.cpp)
add_dependencies(test_molrs_capi molrs_capi_build)
target_link_libraries(test_molrs_capi
    molrs_capi
    GTest::gtest_main
)

enable_testing()
add_test(NAME molrs_capi_tests COMMAND test_molrs_capi)
