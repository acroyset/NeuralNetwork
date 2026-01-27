cmake_minimum_required(VERSION 3.31)
project(NeuralNetwork)

set(CMAKE_CXX_STANDARD 20)

list(PREPEND CMAKE_PREFIX_PATH "/opt/homebrew/opt/sfml@2")
find_package(SFML 2.6 COMPONENTS system window graphics REQUIRED)

add_executable(NeuralNetwork
        src/main.cpp
        src/NeuralNetwork/NeuralNetwork.cpp
        src/NeuralNetwork/Training.cpp
        src/NeuralNetwork/Serialization.cpp
)

target_link_libraries(NeuralNetwork PRIVATE sfml-graphics sfml-window sfml-system)