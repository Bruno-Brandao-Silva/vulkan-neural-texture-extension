add_library(vntx_layer SHARED
    src/config.cpp
    src/layer.cpp
    src/filter.cpp
    src/hooks.cpp
    src/spirv_rewriter.cpp
)

target_include_directories(vntx_layer
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

target_link_libraries(vntx_layer
    PRIVATE
        Vulkan::Headers
)

set_target_properties(vntx_layer PROPERTIES
    OUTPUT_NAME "vntx_layer"
    PREFIX "lib"
    CXX_VISIBILITY_PRESET default
    VISIBILITY_INLINES_HIDDEN OFF
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/manifest/vntx_layer.json.in
    ${CMAKE_BINARY_DIR}/vntx_layer.json
    COPYONLY
)
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/manifest/vntx_layer.json.in
    ${CMAKE_BINARY_DIR}/manifest/vntx_layer.json
    COPYONLY
)

install(TARGETS vntx_layer DESTINATION lib)
install(FILES ${CMAKE_BINARY_DIR}/vntx_layer.json DESTINATION share/vulkan/implicit_layer.d)

