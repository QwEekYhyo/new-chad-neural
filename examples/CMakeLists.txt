file(GLOB examples "${CMAKE_CURRENT_SOURCE_DIR}/*.c")

foreach(example_src ${examples})
    get_filename_component(example_bin ${example_src} NAME_WE)
    add_executable(${example_bin} ${example_src}) 
    target_link_libraries(${example_bin} new_chad_neural)
endforeach()
