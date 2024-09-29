enable_testing()

add_executable(test test_main.c)

target_link_libraries(test new_chad_neural)

add_test(NAME RunMain COMMAND test)
