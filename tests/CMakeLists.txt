enable_testing()

file(GLOB_RECURSE SOURCES *.c)

foreach(SOURCE ${SOURCES})
    get_filename_component(FILE_NAME ${SOURCE} NAME_WE)

    add_executable(${FILE_NAME} ${SOURCE})
    target_link_libraries(${FILE_NAME} new_chad_neural)

    add_test(NAME "Run${FILE_NAME}" COMMAND ${FILE_NAME})
endforeach()
