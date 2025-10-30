cmake_minimum_required(VERSION 3.10)
project(chatBot)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

add_executable(chatBot
    main.cpp
    tensor.h
    linear.h
    transformer.h
    dataset.h
    dataset.cpp
    utils.h
)