#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Training.h"
#include "NeuralNetwork/Serialization.h"
#include <iostream>
#include <vector>
#include <cmath>

std::vector<float> inputs = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
std::vector<std::vector<float>> outputs = {
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
        float input = inputs[i] / 16.0f;
        std::vector<float> networkOut = network.forward({input});
        std::vector<float> expected = outputs[i];

        for (int j = 0; j < 4; j++) {
            totalError += std::abs(networkOut[j] - expected[j]);
        }
    }

    // Return positive fitness (lower error = higher fitness)
    return 64.0f - totalError;  // Max is 64 (16 samples * 4 bits)
}

int main() {
    std::vector<ActivationFunction*> activations = {
        new ReLU(),
        new ReLU(),
        new ReLU(),
        new ReLU(),
        new Sigmoid()
    };

    NeuralNetwork network(1, 4, {128, 64, 32, 16}, activations);

    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        network.getLayer(i).heInitialize();
    }

    // Configure training
    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GENETIC;
    settings.populationSize = 128;
    settings.generations = 5000;
    settings.mutationRate = 0.3f;
    settings.mutationStdDev = 0.8f;
    settings.mutationDecay = 0.0f;
    settings.crossoverRate = 0.8f;
    settings.elitePercent = 0.1f;
    settings.verbose = true;
    settings.logInterval = 500;
    settings.targetFitness = 64.0f;

    // Train
    NetworkTrainer trainer;
    auto result = trainer.train(network, evalFunction, settings);

    const auto& bestNet = trainer.getBestNetwork();
    for (int i = 0; i < 16; i++) {
        auto output = bestNet.forward({float(i)/16.0f});
        std::cout << i << " -> ";
        for (int j = 0; j < 4; j++) {
            std::cout << output[j] << " ";
        }
        std::cout << std::endl;
    }

    NetworkSerializer::saveJSON(bestNet, "Models/1-4_BinarySolver_01-01-26.json");

    return 0;
}