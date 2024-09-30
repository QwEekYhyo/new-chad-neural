enable_testing()

file(GLOB_RECURSE SOURCES *.c)

set(SOURCE_FILE ${CMAKE_SOURCE_DIR}/tests/test_vector.ncn)
set(DEST_FILE ${CMAKE_BINARY_DIR}/tests/test_vector.ncn)

add_custom_command(
    OUTPUT ${DEST_FILE}
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE} ${DEST_FILE}
    DEPENDS ${SOURCE_FILE}
    COMMENT "Copying data files needed by the tests to build directory"
)

add_custom_target(copy_file ALL DEPENDS ${DEST_FILE})

foreach(SOURCE ${SOURCES})
    get_filename_component(FILE_NAME ${SOURCE} NAME_WE)

    add_executable(${FILE_NAME} ${SOURCE})
    target_link_libraries(${FILE_NAME} new_chad_neural)

    add_test(NAME "Run${FILE_NAME}" COMMAND ${FILE_NAME})
endforeach()

add_dependencies(test_load_vector copy_file)
