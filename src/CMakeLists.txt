file(GLOB_RECURSE SOURCES *.c)
file(GLOB_RECURSE HEADERS ../include/*.h)

add_library(new_chad_neural STATIC ${SOURCES} ${HEADERS})

IF (WIN32)
	target_link_libraries(new_chad_neural)
else()
	target_link_libraries(new_chad_neural m)
endif()

target_include_directories(new_chad_neural PUBLIC ../include)
