cmake_minimum_required(VERSION 3.10)

project(mnist-test LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

link_directories(../build/src)

include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}
)

set(SOURCES
    ${CMAKE_SOURCE_DIR}/main.c
    ${CMAKE_SOURCE_DIR}/mnist_utils.c
)

add_executable(mnist-test ${SOURCES})
target_link_libraries(mnist-test new_chad_neural m)
