#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Training.h"
#include "NeuralNetwork/Serialization.h"
#include <iostream>
#include <vector>
#include <cmath>

// Example 1: Simple XOR Problem
class XORProblem {
public:
    static float evaluateNetwork(const NeuralNetwork& net) {
        // XOR training data
        std::vector<std::pair<std::vector<float>, std::vector<float>>> data = {
            {{0.0f, 0.0f}, {0.0f}},
            {{0.0f, 1.0f}, {1.0f}},
            {{1.0f, 0.0f}, {1.0f}},
            {{1.0f, 1.0f}, {0.0f}}
        };

        float totalError = 0.0f;
        for (const auto& [input, target] : data) {
            auto output = net.forward(input);
            float error = std::abs(output[0] - target[0]);
            totalError += error;
        }

        // Return inverted error as fitness (lower error = higher fitness)
        return 4.0f - totalError;  // Perfect score is 4.0
    }
};

// Example 2: Function Approximation (sin function)
class SinApproximation {
public:
    static float evaluateNetwork(const NeuralNetwork& net) {
        float totalError = 0.0f;

        // Test on 20 points
        for (int i = 0; i < 20; ++i) {
            float x = (float(i) / 19.0f) * 2.0f * 3.14159f;
            auto output = net.forward({x});
            float expectedOutput = (std::sin(x) + 1.0f) / 2.0f;  // Normalize to [0, 1]
            float error = std::abs(output[0] - expectedOutput);
            totalError += error;
        }

        return 20.0f - totalError;
    }
};

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         Neural Network Library - Comprehensive Examples    ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";

    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "EXAMPLE 1: XOR Problem Training\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Create network: 2 inputs -> 4 hidden -> 1 output
    std::vector<ActivationFunction*> activations = {
        new ReLU(),      // Hidden layer activation
        new Sigmoid()    // Output layer activation
    };

    NeuralNetwork network(2, 1, {4}, activations);

    // Configure training
    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GENETIC;
    settings.populationSize = 50;
    settings.generations = 1000;
    settings.mutationRate = 0.2f;
    settings.mutationStdDev = 0.5f;
    settings.verbose = true;
    settings.logInterval = 100;
    settings.targetFitness = 4.0f;  // Very good XOR solution

    // Train
    NetworkTrainer trainer;
    auto result = trainer.train(network, XORProblem::evaluateNetwork, settings);

    // Evaluate on test data
    std::cout << "\n\nFinal XOR Test:\n";
    std::vector<std::pair<std::vector<float>, std::string>> tests = {
        {{0.0f, 0.0f}, "0 XOR 0 should be ~0: "},
        {{0.0f, 1.0f}, "0 XOR 1 should be ~1: "},
        {{1.0f, 0.0f}, "1 XOR 0 should be ~1: "},
        {{1.0f, 1.0f}, "1 XOR 1 should be ~0: "}
    };

    const auto& bestNet = trainer.getBestNetwork();
    for (const auto& [input, label] : tests) {
        auto output = bestNet.forward(input);
        std::cout << label << output[0] << "\n";
    }

    NetworkSerializer::saveJSON(bestNet, "Models/XOR_Genetic_01-01-26.json");

    return 0;
}