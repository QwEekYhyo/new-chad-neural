enable_testing()

add_executable(test_main test_main.c)

target_link_libraries(test_main new_chad_neural)

add_test(NAME RunMain COMMAND test_main)
