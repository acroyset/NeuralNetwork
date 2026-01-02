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
    GENETIC,           // Genetic algorithm
    NEUROEVOLUTION,    // Neuroevolution (NEAT-like)
    RANDOM_SEARCH,     // Random search with noise
    GRADIENT_DESCENT   // Gradient descent with backpropagation
};

struct TrainingSettings {
    // Algorithm configuration
    TrainingAlgorithm algorithm = TrainingAlgorithm::GENETIC;

    // Population-based settings
    uint32_t populationSize = 50;
    uint32_t generations = 100;
    float mutationRate = 0.1f;           // Probability of mutation per weight
    float mutationStdDev = 0.1f;         // Standard deviation of mutation
    float mutationDecay = 0.01f;         // Percent of StdDev decay per generation
    float crossoverRate = 0.7f;          // Portion of population that reproduces
    float elitePercent = 0.1f;           // Percent of population that are considered Elite and stay

    // Neuroevolution specific
    uint32_t topSpecimens = 10;          // Number of top performers to keep
    float noiseScale = 0.15f;            // Noise scale for exploration

    // Gradient descent specific
    float learningRate = 0.01f;          // Learning rate (alpha)
    float learningRateDecay = 0.0f;      // Decay per generation (0 = no decay)
    uint32_t batchSize = 16;             // Mini-batch size (or 0 for full dataset)
    TrainingData* trainingData = nullptr;// Required for GRADIENT_DESCENT, ignored for other algorithms

    // Training behavior
    bool verbose = true;                 // Print progress information
    uint32_t logInterval = 10;           // Log every N generations
    uint32_t saveInterval = 50;          // Save network every N generations
    std::string checkpointPath;          // Path to save checkpoints (empty = no saving)

    // Termination conditions
    float targetFitness = 1000.0f;       // Stop training if fitness reaches this

    // Random seed
    uint32_t randomSeed = 0;             // 0 = use system time

    // Multithreading
    bool enableMultithreading = false;   // Toggle parallelization
    uint32_t numThreads = 0;             // 0 = auto-detect

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

    void trainGradientDescent(NeuralNetwork& network,
                          const RewardFunction& reward,
                          const TrainingSettings& settings);

    // Helper functions
    float evaluateNetwork(const NeuralNetwork& network, const RewardFunction& reward);
    static void logProgress(uint32_t generation, float best, float avg,
                    const TrainingSettings& settings) ;

    void evaluatePopulationParallel(std::vector<NeuralNetwork>& population,
                                    std::vector<float>& fitness,
                                    const RewardFunction& reward,
                                    const TrainingSettings& settings,
                                    uint32_t startIdx);

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
