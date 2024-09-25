cmake_minimum_required(VERSION 3.10)

project(new_chad_neural LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

file(GLOB_RECURSE SOURCES src/*.c)
file(GLOB_RECURSE HEADERS include/*.h)

include_directories(include)

add_executable(new_chad_neural ${SOURCES} ${HEADERS})
target_link_libraries(new_chad_neural m)

if(MSVC)
    target_compile_options(new_chad_neural PRIVATE /W4 /WX)
else()
    target_compile_options(new_chad_neural PRIVATE -Wall -Wextra -Wpedantic -O3)
endif()
