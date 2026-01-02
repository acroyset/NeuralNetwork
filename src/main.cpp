#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Training.h"
#include "NeuralNetwork/Serialization.h"
#include <iostream>
#include <vector>

std::vector<std::vector<float>> outputs = {{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}, {8}, {9}, {10}, {11}, {12}, {13}, {14}, {15}};
std::vector<std::vector<float>> inputs = {
    {0,0,0,0},
    {0,0,0,1},
    {0,0,1,0},
    {0,0,1,1},
    {0,1,0,0},
    {0,1,0,1},
    {0,1,1,0},
    {0,1,1,1},
    {1,0,0,0},
    {1,0,0,1},
    {1,0,1,0},
    {1,0,1,1},
    {1,1,0,0},
    {1,1,0,1},
    {1,1,1,0},
    {1,1,1,1},
};

float evalFunction(const NeuralNetwork& network) {
    float totalError = 0.0f;

    for (int i = 0; i < 16; i++) {
        const std::vector<float>& input = inputs[i];
        float networkOut = network.forward(input)[0];
        float expected = outputs[i][0];

        totalError += abs(networkOut-expected);
    }

    // Return positive fitness (lower error = higher fitness)
    return 64.0f - totalError;  // Max is 64 (16 samples * 4 bits)
}

int main() {
    std::vector<ActivationFunction*> activations = {
        new ReLU(),
        new ReLU(),
        new Sigmoid()
    };

    NeuralNetwork network(1, 4, {16, 8}, activations);

    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        network.getLayer(i).heInitialize();
    }

    // Configure training
    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GRADIENT_DESCENT;
    settings.generations = 10000;

    settings.logInterval = 200;
    settings.targetFitness = 63.999999f;
    settings.enableMultithreading = false;
    settings.batchSize = 8;

    TrainingData data;
    data.inputs = outputs;
    data.outputs = inputs;
    settings.trainingData = &data;

    // Train
    NetworkTrainer trainer;
    auto result = trainer.train(network, evalFunction, settings);

    std::cout << std::endl;
    const auto& bestNet = trainer.getBestNetwork();
    for (int i = 0; i < 16; i++) {
        auto output = bestNet.forward(inputs[i]);
        std::cout << i << " -> " << output[0] << std::endl;
    }

    NetworkSerializer::saveJSON(bestNet, "Models/4-1_BinarySolver_01-01-26.json");

    return 0;
}