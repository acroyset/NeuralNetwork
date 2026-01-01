//
// Created by Andreas Royset on 1/1/26.
//

#ifndef TRAINING_H
#define TRAINING_H

#pragma once

#include "neuralNetwork.h"
#include <functional>
#include <chrono>
#include <vector>


using RewardFunction = std::function<float(const NeuralNetwork&)>;

enum class TrainingAlgorithm {
    GENETIC,           // Genetic algorithm
    NEUROEVOLUTION,    // Neuroevolution (NEAT-like)
    RANDOM_SEARCH      // Random search with noise
};

struct TrainingSettings {
    // Algorithm configuration
    TrainingAlgorithm algorithm = TrainingAlgorithm::GENETIC;

    // Population-based settings
    uint32_t populationSize = 50;
    uint32_t generations = 100;
    float mutationRate = 0.1f;           // Probability of mutation per weight
    float mutationStdDev = 0.1f;         // Standard deviation of mutation
    float crossoverRate = 0.7f;          // Portion of population that reproduces

    // Neuroevolution specific
    uint32_t topSpecimens = 10;          // Number of top performers to keep
    float noiseScale = 0.15f;            // Noise scale for exploration

    // Training behavior
    bool verbose = true;                 // Print progress information
    uint32_t logInterval = 10;           // Log every N generations
    uint32_t saveInterval = 50;          // Save network every N generations
    std::string checkpointPath;     // Path to save checkpoints (empty = no saving)

    // Termination conditions
    float targetFitness = 1000.0f;       // Stop training if fitness reaches this
    uint32_t maxEvaluations = 1000000;   // Maximum network evaluations
    std::chrono::seconds timeLimit{3600}; // One hour time limit

    // Random seed
    uint32_t randomSeed = 0;             // 0 = use system time

    // Validation
    void validate() const;
};

struct TrainingResult {
    float bestFitness = 0.0f;
    float averageFitness = 0.0f;
    float worstFitness = 0.0f;
    uint32_t generationsTrained = 0;
    uint32_t totalEvaluations = 0;
    std::chrono::milliseconds trainingTime{0};
    std::vector<float> fitnessHistory;
};

class NetworkTrainer {
    NeuralNetwork bestNetwork;
    TrainingResult result;
    uint32_t totalEvaluations;

    // Algorithm implementations
    void trainGenetic(NeuralNetwork& network,
                     const RewardFunction& reward,
                     const TrainingSettings& settings);

    void trainNeuroevolution(const NeuralNetwork& network,
                            const RewardFunction& reward,
                            const TrainingSettings& settings);

    void trainRandomSearch(const NeuralNetwork& network,
                          const RewardFunction& reward,
                          const TrainingSettings& settings);

    // Helper functions
    float evaluateNetwork(const NeuralNetwork& network, const RewardFunction& reward);
    void logProgress(uint32_t generation, float best, float avg,
                    const TrainingSettings& settings) const;

public:

    NetworkTrainer(): bestNetwork(NeuralNetwork(1, 1, {}, {new ReLU})), totalEvaluations(0) {}

    TrainingResult train(NeuralNetwork& network,
                         const RewardFunction& reward,
                         const TrainingSettings& settings);

    [[nodiscard]] const NeuralNetwork& getBestNetwork() const {
        return bestNetwork;
    }

    [[nodiscard]] const TrainingResult& getResult() const {
        return result;
    }
};



#endif //TRAINING_H
