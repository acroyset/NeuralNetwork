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
        std::vector<float> networkOut = network.forward(input);
        const std::vector<float>& expected = outputs[i];

        float error = 0;

        for (int j = 0; j < networkOut.size(); j++) {
            error += abs(networkOut[j]-expected[j]);
        }

        totalError += error;
    }

    // Return positive fitness (lower error = higher fitness)
    return 64.0f - totalError;  // Max is 64 (16 samples * 4 bits)
}

int main() {
    for (auto& input : inputs) {
        input[0] /= 15.0f;  // Normalize 0-15 to 0-1
    }

    std::vector<ActivationFunction*> activations = {
        new Tanh(),      // Better for small single input
        new Tanh(),
        new Tanh(),
        new Sigmoid()    // For binary outputs [0,1]
    };

    // Larger hidden layers to memorize the mapping
    NeuralNetwork network(1, 4, {64, 64, 32}, activations);

    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        network.getLayer(i).heInitialize();
    }

    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GRADIENT_DESCENT;
    settings.generations = 10000;
    settings.learningRate = 0.05f;
    settings.decayRate = 0.0001f;
    settings.logInterval = 500;
    settings.batchSize = 4;

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
        auto output = bestNet.forward(outputs[i]);
        std::cout << i << " -> ";
        for (float j : inputs[i]) {
            std::cout << j << " ";
        }
        std::cout << std::endl;
    }

    NetworkSerializer::saveJSON(bestNet, "Models/4-1_BinarySolver_01-01-26.json");

    return 0;
}