cmake_minimum_required(VERSION 3.14)
project(bsm_iv)

set(CMAKE_CXX_STANDARD 11)

option(WITH_TESTS "build fetch GTest and Google benchmark and compile targets for them")

add_library(bsm bsm.cpp)

add_executable(main main.cpp)

target_link_libraries(main bsm)

if(WITH_TESTS)
    include(FetchContent)
    
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG e2239ee6043f73722e7aa812a459f54a28552929  # v1.11.0
    )
    FetchContent_MakeAvailable(googletest)

    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG 0d98dba29d66e93259db7daa53a9327df767a415  # v1.6.1
    )
    FetchContent_MakeAvailable(benchmark)

    add_executable(performance_tests tests/performance.cpp)
    target_link_libraries(performance_tests bsm benchmark::benchmark)

    enable_testing()

    add_executable(unit_tests tests/unit.cpp)
    target_link_libraries(unit_tests bsm gtest_main)

    include(GoogleTest)
    gtest_discover_tests(unit_tests)
endif()

