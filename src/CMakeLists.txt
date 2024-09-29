file(GLOB_RECURSE SOURCES *.c)
file(GLOB_RECURSE HEADERS ../include/*.h)

add_library(new_chad_neural STATIC ${SOURCES} ${HEADERS})

target_link_libraries(new_chad_neural m)

target_include_directories(new_chad_neural PUBLIC ../include)
