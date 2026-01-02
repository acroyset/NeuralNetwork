//
// Created by Andreas Royset on 1/1/26.
//

#include "Training.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>


void TrainingSettings::validate() const {
    if (populationSize < 2) {
        throw std::invalid_argument("Population size must be at least 2");
    }
    if (generations == 0) {
        throw std::invalid_argument("Generations must be greater than 0");
    }
    if (mutationRate < 0.0f || mutationRate > 1.0f) {
        throw std::invalid_argument("Mutation rate must be between 0 and 1");
    }
    if (crossoverRate < 0.0f || crossoverRate > 1.0f) {
        throw std::invalid_argument("Crossover rate must be between 0 and 1");
    }
}

float NetworkTrainer::evaluateNetwork(const NeuralNetwork& network,
                                      const RewardFunction& reward) {
    totalEvaluations++;
    return reward(network);
}

void NetworkTrainer::logProgress(const uint32_t generation, const float best, const float avg,
                                const TrainingSettings& settings) const {
    if (!settings.verbose) return;

    float decay = pow(1-settings.mutationDecay, generation);
    bool isGenetic = settings.algorithm == TrainingAlgorithm::GENETIC;

    std::cout << "Gen " << std::setw(5) << generation
              << " | Best: " << std::setw(10) << std::fixed << std::setprecision(4) << best
              << " | Avg: " << std::setw(10) << std::setprecision(4) << avg
              << " | " << (isGenetic ? "Mutation: " : "Noise Scale: ") << std::setw(10) << std::setprecision(4)
              << (isGenetic ? settings.mutationStdDev*decay : settings.noiseScale*decay)
              << std::endl;
}

TrainingResult NetworkTrainer::train(NeuralNetwork& network,
                                     const RewardFunction& reward,
                                     const TrainingSettings& settings) {
    // Validate settings
    settings.validate();

    // Initialize
    auto startTime = std::chrono::high_resolution_clock::now();
    totalEvaluations = 0;
    result = TrainingResult();
    bestNetwork = network.clone();

    // Set random seed
    if (settings.randomSeed != 0) {
        srand(settings.randomSeed);
    }

    if (settings.verbose) {
        std::cout << "Starting training with " << settings.populationSize << " population\n"
                  << "Algorithm: ";

        switch (settings.algorithm) {
            case TrainingAlgorithm::GENETIC:
                std::cout << "Genetic Algorithm\n";
                break;
            case TrainingAlgorithm::NEUROEVOLUTION:
                std::cout << "Neuroevolution\n";
                break;
            case TrainingAlgorithm::RANDOM_SEARCH:
                std::cout << "Random Search\n";
                break;
        }
        std::cout << std::string(60, '-') << std::endl;
    }

    // Run appropriate algorithm
    switch (settings.algorithm) {
        case TrainingAlgorithm::GENETIC:
            trainGenetic(network, reward, settings);
            break;
        case TrainingAlgorithm::NEUROEVOLUTION:
            trainNeuroevolution(network, reward, settings);
            break;
        case TrainingAlgorithm::RANDOM_SEARCH:
            trainRandomSearch(network, reward, settings);
            break;
    }

    // Record results
    auto endTime = std::chrono::high_resolution_clock::now();
    result.trainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(
        endTime - startTime
    );
    result.totalEvaluations = totalEvaluations;
    network = bestNetwork.clone();

    if (settings.verbose) {
        std::cout << std::string(60, '-') << "\n"
                  << "Training complete!\n"
                  << "Best fitness: " << result.bestFitness << "\n"
                  << "Total evaluations: " << result.totalEvaluations << "\n"
                  << "Training time: " << result.trainingTime.count() << " ms\n";
    }

    return result;
}

void NetworkTrainer::trainGenetic(NeuralNetwork& network,
                                  const RewardFunction& reward,
                                  const TrainingSettings& settings) {
    std::mt19937 rng(settings.randomSeed ? settings.randomSeed : std::random_device{}());
    std::uniform_real_distribution mutationProb(0.0f, 1.0f);

    // Initialize population
    std::vector<NeuralNetwork> population;
    std::vector<float> fitness;

    population.reserve(settings.populationSize);
    fitness.reserve(settings.populationSize);

    float mutationStdDev = settings.mutationStdDev;

    for (uint32_t i = 0; i < settings.populationSize; ++i) {
        population.push_back(network.clone());
        if (i > 0) {
            population[i].addNoise(mutationStdDev);
        }
    }

    // Evaluate initial population
    for (auto& individual : population) {
        fitness.push_back(evaluateNetwork(individual, reward));
    }

    // Main loop
    for (uint32_t gen = 0; gen < settings.generations; ++gen) {
        mutationStdDev *= 1-settings.mutationDecay;
        // Sort by fitness
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::ranges::sort(indices,
                          [&fitness](size_t a, size_t b) { return fitness[a] > fitness[b]; });

        // Track best
        float bestFit = fitness[indices[0]];
        float avgFit = std::accumulate(fitness.begin(), fitness.end(), 0.0f) / float(fitness.size());

        if (bestFit > result.bestFitness) {
            result.bestFitness = bestFit;
            bestNetwork = population[indices[0]].clone();
        }

        result.fitnessHistory.push_back(bestFit);

        if (gen % settings.logInterval == 0) {
            logProgress(gen, bestFit, avgFit, settings);
        }

        // Check termination conditions
        if (bestFit >= settings.targetFitness ||
            totalEvaluations >= settings.maxEvaluations) {
            result.generationsTrained = gen + 1;
            return;
        }

        // Create next generation
        std::vector<NeuralNetwork> nextGen;
        std::vector<float> nextFitness;

        // Elitism: keep top performers
        uint32_t eliteSize = std::max(1u, uint32_t(float(settings.populationSize) * settings.elitePercent));
        for (uint32_t i = 0; i < eliteSize && i < indices.size(); ++i) {
            nextGen.push_back(population[indices[i]].clone());
            nextFitness.push_back(fitness[indices[i]]);
        }

        // Reproduction and mutation
        size_t crossoverAmount = float(settings.populationSize) * settings.crossoverRate;
        std::uniform_int_distribution<size_t> parentSelection(0, std::min(crossoverAmount, indices.size() - 1));

        while (nextGen.size() < settings.populationSize) {
            // Select parents
            size_t parent1Idx = indices[parentSelection(rng)];
            size_t parent2Idx = indices[parentSelection(rng)];

            // Create offspring
            NeuralNetwork offspring = population[parent1Idx].clone();

            // Crossover: blend with parent2
            float crossoverBlend = 0.5f;
            for (size_t layer = 0; layer < offspring.getLayerCount(); ++layer) {
                auto& offWeights = offspring.getLayer(layer).getWeights();
                const auto& p2Weights = population[parent2Idx].getLayer(layer).getWeights();

                for (size_t i = 0; i < offWeights.size(); ++i) {
                    for (size_t j = 0; j < offWeights[i].size(); ++j) {
                        offWeights[i][j] = crossoverBlend * offWeights[i][j] +
                                          (1.0f - crossoverBlend) * p2Weights[i][j];
                    }
                }

                auto& offBiases = offspring.getLayer(layer).getBiases();
                const auto& p2Biases = population[parent2Idx].getLayer(layer).getBiases();

                for (size_t i = 0; i < offBiases.size(); ++i) {
                    offBiases[i] = crossoverBlend * offBiases[i] +
                                  (1.0f - crossoverBlend) * p2Biases[i];
                }
            }

            // Mutation: apply to each weight with probability
            for (size_t layer = 0; layer < offspring.getLayerCount(); ++layer) {
                auto& weights = offspring.getLayer(layer).getWeights();
                auto& biases = offspring.getLayer(layer).getBiases();

                // Mutate weights
                for (auto& row : weights) {
                    for (auto& w : row) {
                        if (mutationProb(rng) < settings.mutationRate) {
                            std::normal_distribution<float> mutationDist(0.0f, mutationStdDev);
                            w += mutationDist(rng);
                        }
                    }
                }

                // Mutate biases
                for (auto& b : biases) {
                    if (mutationProb(rng) < settings.mutationRate) {
                        std::normal_distribution<float> mutationDist(0.0f, mutationStdDev);
                        b += mutationDist(rng);
                    }
                }
            }

            float fit = evaluateNetwork(offspring, reward);
            nextFitness.push_back(fit);

            nextGen.push_back(std::move(offspring));

            if (totalEvaluations >= settings.maxEvaluations) break;
        }

        population = nextGen;
        fitness = nextFitness;
    }

    result.generationsTrained = settings.generations;
}

void NetworkTrainer::trainNeuroevolution(const NeuralNetwork& network,
                                         const RewardFunction& reward,
                                         const TrainingSettings& settings) {
    std::mt19937 rng(settings.randomSeed ? settings.randomSeed : std::random_device{}());

    // Initialize population with noise
    std::vector<NeuralNetwork> population;
    std::vector<float> fitness;

    float noiseScale = settings.noiseScale;

    for (uint32_t i = 0; i < settings.populationSize; ++i) {
        auto individual = network.clone();
        if (i > 0) {
            individual.addNoise(noiseScale);
        }
        population.push_back(individual);
        fitness.push_back(evaluateNetwork(individual, reward));
    }

    // Main loop
    for (uint32_t gen = 0; gen < settings.generations; ++gen) {
        noiseScale *= 1-settings.mutationDecay;
        // Sort by fitness
        std::vector<size_t> indices(population.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::ranges::sort(indices,
                          [&fitness](size_t a, size_t b) { return fitness[a] > fitness[b]; });

        float bestFit = fitness[indices[0]];
        float avgFit = std::accumulate(fitness.begin(), fitness.end(), 0.0f) / float(fitness.size());

        if (bestFit > result.bestFitness) {
            result.bestFitness = bestFit;
            bestNetwork = population[indices[0]].clone();
        }

        result.fitnessHistory.push_back(bestFit);

        if (gen % settings.logInterval == 0) {
            logProgress(gen, bestFit, avgFit, settings);
        }

        // Check termination
        if (bestFit >= settings.targetFitness ||
            totalEvaluations >= settings.maxEvaluations) {
            result.generationsTrained = gen + 1;
            return;
        }

        // Keep top specimens and mutate
        std::vector<NeuralNetwork> nextGen;
        std::vector<float> nextFitness;

        uint32_t topCount = std::min(settings.topSpecimens, uint32_t(indices.size()));
        for (uint32_t i = 0; i < topCount; ++i) {
            nextGen.push_back(population[indices[i]].clone());
            nextFitness.push_back(fitness[indices[i]]);
        }

        // Generate offspring from top specimens
        while (nextGen.size() < settings.populationSize) {
            size_t parentIdx = indices[rng() % topCount];
            auto offspring = population[parentIdx].clone();
            offspring.addNoise(noiseScale);

            nextGen.push_back(offspring);
            float fit = evaluateNetwork(offspring, reward);
            nextFitness.push_back(fit);

            if (totalEvaluations >= settings.maxEvaluations) break;
        }

        population = nextGen;
        fitness = nextFitness;
    }

    result.generationsTrained = settings.generations;
}

void NetworkTrainer::trainRandomSearch(const NeuralNetwork& network,
                                      const RewardFunction& reward,
                                      const TrainingSettings& settings) {

    // Evaluate initial network
    float bestFit = evaluateNetwork(network, reward);
    result.bestFitness = bestFit;
    bestNetwork = network.clone();

    float noiseScale = settings.noiseScale;

    // Random search iterations
    uint32_t iterations = 0;
    while (iterations < settings.generations && totalEvaluations < settings.maxEvaluations) {
        noiseScale *= 1-settings.mutationDecay;
        // Create random perturbation
        auto candidate = bestNetwork.clone();
        candidate.addNoise(noiseScale);

        float fit = evaluateNetwork(candidate, reward);

        if (fit > bestFit) {
            bestFit = fit;
            result.bestFitness = fit;
            bestNetwork = candidate.clone();
        }

        result.fitnessHistory.push_back(bestFit);

        if (iterations % settings.logInterval == 0) {
            logProgress(iterations, bestFit, bestFit, settings);
        }

        if (bestFit >= settings.targetFitness) {
            result.generationsTrained = iterations + 1;
            return;
        }

        iterations++;
    }

    result.generationsTrained = iterations;
}
