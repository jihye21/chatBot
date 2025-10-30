cmake_minimum_required(VERSION 3.10)
project(chatBot)

set(CMAKE_CXX_STANDARD 17)

set(SOURCES main.cpp)
include_directories(${PROJECT_SOURCE_DIR})
add_executable(chatBot ${SOURCES})
