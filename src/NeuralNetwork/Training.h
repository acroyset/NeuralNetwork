//
// Created by Andreas Royset on 1/1/26.
//

#ifndef TRAINING_H
#define TRAINING_H

#include "NeuralNetwork.h"
#include <functional>
#include <chrono>
#include <vector>
#include <condition_variable>

using RewardFunction = std::function<float(const NeuralNetwork&)>;

struct TrainingData {
    std::vector<std::vector<float>> inputs;
    std::vector<std::vector<float>> outputs;
};

enum class TrainingAlgorithm {
    NONE,              // Default
    GENETIC,           // Genetic algorithm
    NEUROEVOLUTION,    // Neuroevolution (NEAT-like)
    RANDOM_SEARCH,     // Random search with noise
    GRADIENT_DESCENT   // Gradient descent with backpropagation
};

struct TrainingSettings {
    TrainingAlgorithm algorithm = TrainingAlgorithm::NONE; // Algorithm configuration

    uint32_t populationSize = 0; // Number of Networks tested per generation (GENETIC, NEUROEVOLUTION)

    uint32_t generations = 0; // Number of generations tested

    float noiseStdDev = 0.1f; // Amount of randomness added to weights (GENETIC, NEUROEVOLUTION, RANDOM_SEARCH)
    float crossoverRate = 0.7f; // % of population that reproduces (GENETIC)
    float topPercentage = 0.1f; // % of population that are considered best and stay (GENETIC, NEUROEVOLUTION)

    float decayRate = 0.0f; // Decay % per generation (0 = no decay)

    float learningRate = 0.01f; // Learning rate (alpha) (GRADIENT_DESCENT)
    uint32_t batchSize = 0; // Multithreading batch size (GENETIC, NEUROEVOLUTION)   Mini batch size (GRADIENT_DESCENT)
    TrainingData* trainingData = nullptr; // Training Date (GRADIENT_DESCENT)

    bool verbose = true; // Enable progress reports

    uint32_t logInterval = 50; // Progress report interval

    bool targetPerformanceEnable = false; // Stop if fitness reaches targetPerformance
    float targetPerformance = 0.0f; // Stop training if performance reaches this

    uint32_t randomSeed = 0; // Random Seed (0 = use system time)

    bool enableMultithreading = true; // Toggle parallelization (GENETIC, NEUROEVOLUTION, RANDOM_SEARCH)
    uint32_t numThreads = 0; // Number of available threads (0 = auto-detect)

    void validate() const;
};

struct TrainingResult {
    float bestFitness = 0.0f;
    float bestRMSE = float(pow(2, 64));
    float averageFitness = 0.0f;
    float worstFitness = 0.0f;
    uint32_t generationsTrained = 0;
    uint32_t totalEvaluations = 0;
    std::chrono::milliseconds trainingTime{0};
    std::vector<float> history;
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

    void trainGradientDescent(NeuralNetwork& network,
                          const RewardFunction& reward,
                          const TrainingSettings& settings);

    // Helper functions
    float evaluateNetwork(const NeuralNetwork& network, const RewardFunction& reward);
    static void logProgress(const TrainingResult& result, const TrainingSettings& settings) ;

    void evaluatePopulationParallel(const std::vector<NeuralNetwork>& population,
                                    std::vector<float>& fitness,
                                    const RewardFunction& reward,
                                    const TrainingSettings& settings,
                                    uint32_t skipIdx);

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
