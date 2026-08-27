find_library(VULKAN_LOADER_LIB NAMES vulkan vulkan-1 libvulkan.so.1 PATHS /usr/lib /usr/lib/x86_64-linux-gnu /usr/local/lib)

add_executable(ntc_headless_test
    headless/test_main.cpp
    headless/test_header_layout.cpp
    headless/test_filter.cpp
    headless/test_vulkan_interception.cpp
    headless/test_visual_quality.cpp
    headless/test_fallback.cpp
    headless/test_stress_boundary.cpp
    headless/test_wsl2_lavapipe.cpp
    headless/test_guardrail.cpp
)

target_include_directories(ntc_headless_test
    PRIVATE
        ${CMAKE_SOURCE_DIR}/layer/include
)

target_link_libraries(ntc_headless_test
    PRIVATE
        vntx_layer
        Vulkan::Headers
        GTest::gtest
        ${CMAKE_DL_LIBS}
        pthread
)

if(VULKAN_LOADER_LIB)
    target_link_libraries(ntc_headless_test PRIVATE ${VULKAN_LOADER_LIB})
    target_compile_definitions(ntc_headless_test PRIVATE VNTX_HAS_VULKAN_LOADER=1)
endif()

include(GoogleTest)
gtest_discover_tests(ntc_headless_test
    PROPERTIES
        ENVIRONMENT "VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json"
)
