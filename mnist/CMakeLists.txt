cmake_minimum_required(VERSION 3.10)

project(mnist-test LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

link_directories(../build/src)

# include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/../include)

# set(SOURCES ${CMAKE_SOURCE_DIR}/mnist/main.c)
set(SOURCES ${CMAKE_SOURCE_DIR}/../mnist/main.c)

add_executable(mnist-test ${SOURCES})
target_link_libraries(mnist-test new_chad_neural m)
