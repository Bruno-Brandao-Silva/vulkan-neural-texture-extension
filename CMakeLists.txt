cmake_minimum_required(VERSION 3.25)
project(vulkan_neural_texture_extension
    VERSION 0.1.0
    LANGUAGES C CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)

# Strict warnings
if(MSVC)
    add_compile_options(/W4 /WX /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Sanitizer configuration for debug builds
option(VNTX_ENABLE_SANITIZERS "Enable AddressSanitizer and UndefinedBehaviorSanitizer" ON)
if(VNTX_ENABLE_SANITIZERS AND NOT MSVC)
    add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
    add_link_options(-fsanitize=address,undefined)
endif()

option(VNTX_BUILD_TESTS "Build C++ test suites" ON)

include(FetchContent)

# Fetch official Khronos Vulkan Headers
set(VULKAN_HEADERS_ENABLE_MODULE OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_TESTS OFF CACHE BOOL "" FORCE)
set(VULKAN_HEADERS_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)

FetchContent_Declare(
    VulkanHeaders
    GIT_REPOSITORY https://github.com/KhronosGroup/Vulkan-Headers.git
    GIT_TAG        v1.3.296
    GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(VulkanHeaders)

# Fetch GoogleTest
if(VNTX_BUILD_TESTS)
    enable_testing()
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
        GIT_SHALLOW    TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
endif()

add_subdirectory(layer)

if(VNTX_BUILD_TESTS)
    add_subdirectory(tests)
endif()
