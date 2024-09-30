cmake_minimum_required(VERSION 3.10)

project(new_chad_neural LANGUAGES C)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include_directories(include)

if(MSVC)
    add_compile_options(/W3 /WX /OX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3)
endif()

enable_testing()

add_subdirectory(src)
add_subdirectory(tests)
