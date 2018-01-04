find_package(OpenCV REQUIRED core highgui )

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../src)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/../../src)



include_directories(${CMAKE_CURRENT_BINARY_DIR})

add_executable(yolo yolo.cpp)
target_link_libraries(yolo ncnn ${OpenCV_LIBS})