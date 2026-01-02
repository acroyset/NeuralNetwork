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
    float totalFitness = 0.0f;

    for (int i = 0; i < 16; i++) {
        float input = inputs[i];

        std::vector<float> networkOut = network.forward({input});
        std::vector<float> expected = outputs[i];

        float fitness = 0;

        for (int j = 0; j < 4; j++) {
            fitness -= float(pow(networkOut[j] - expected[j], 2));
        }

        totalFitness += fitness;
    }

    return totalFitness/16;
}

int main() {
    std::vector<ActivationFunction*> activations = {
        new LeakyReLU(),
        new LeakyReLU(),
        new LeakyReLU(),
        new Sigmoid()
    };

    NeuralNetwork network(1, 4, {16, 32, 8}, activations);

    // Configure training
    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GENETIC;
    settings.populationSize = 128;
    settings.generations = 1000;
    settings.mutationRate = 0.1f;
    settings.mutationStdDev = 0.5f;
    settings.mutationDecay = 0.0005f;
    settings.verbose = true;
    settings.logInterval = 100;
    settings.targetFitness = 0.0f;

    // Train
    NetworkTrainer trainer;
    auto result = trainer.train(network, evalFunction, settings);

    const auto& bestNet = trainer.getBestNetwork();
    for (int i = 0; i < 16; i++) {
        auto output = bestNet.forward({float(i)});
        std::cout << i << " -> ";
        for (int j = 0; j < 4; j++) {
            std::cout << output[j] << " ";
        }
        std::cout << std::endl;
    }

    NetworkSerializer::saveJSON(bestNet, "Models/1-4_BinarySolver_01-01-26.json");

    return 0;
}