enable_testing()

file(GLOB_RECURSE SOURCES *.c)

set(SOURCE_FILE1 ${CMAKE_SOURCE_DIR}/tests/test_vector.ncn)
set(DEST_FILE1 ${CMAKE_BINARY_DIR}/tests/test_vector.ncn)

set(SOURCE_FILE2 ${CMAKE_SOURCE_DIR}/tests/test_matrix.ncn)
set(DEST_FILE2 ${CMAKE_BINARY_DIR}/tests/test_matrix.ncn)

add_custom_command(
    OUTPUT ${DEST_FILE1} ${DEST_FILE2}
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE1} ${DEST_FILE1}
    COMMAND ${CMAKE_COMMAND} -E copy ${SOURCE_FILE2} ${DEST_FILE2}
    DEPENDS ${SOURCE_FILE1} ${SOURCE_FILE2}
    COMMENT "Copying data files needed by the tests to build directory"
)
add_custom_target(copy_test_files ALL DEPENDS ${DEST_FILE1} ${DEST_FILE2})

foreach(SOURCE ${SOURCES})
    get_filename_component(FILE_NAME ${SOURCE} NAME_WE)

    add_executable(${FILE_NAME} ${SOURCE})
    target_link_libraries(${FILE_NAME} new_chad_neural)

    add_test(NAME "Run${FILE_NAME}" COMMAND ${FILE_NAME})
endforeach()

add_dependencies(test_load_vector copy_test_files)
add_dependencies(test_load_matrix copy_test_files)
