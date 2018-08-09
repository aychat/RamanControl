cmake_minimum_required (VERSION 2.8.3)
project (RamanAssistedControl)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
link_directories(${PROJECT_SOURCE_DIR})

add_library(RamanAssistedControl_utils SHARED RamanAssistedControl_utils.c)

add_library(RamanAssistedControl SHARED RamanAssistedControl.c)
target_link_libraries(RamanAssistedControl RamanAssistedControl_utils nlopt)
