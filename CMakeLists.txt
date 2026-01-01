cmake_minimum_required(VERSION 3.31)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

add_executable(NeuralNetwork
        src/main.cpp
        src/NeuralNetwork/NeuralNetwork.cpp
        src/NeuralNetwork/Training.cpp
        src/NeuralNetwork/Serialization.cpp
)
