cmake_minimum_required(VERSION 3.10)

project(new_chad_neural LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

add_subdirectory(src)
add_subdirectory(tests)
