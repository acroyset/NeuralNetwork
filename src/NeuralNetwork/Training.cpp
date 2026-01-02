//
// Created by Andreas Royset on 1/1/26.
//

#include "Training.h"
#include <iostream>
#include <algorithm>
#include <random>
#include <numeric>
#include <iomanip>
#include <thread>

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

void NetworkTrainer::logProgress(const TrainingResult& result, const TrainingSettings& settings) {
    if (!settings.verbose) return;

    auto decay = float(pow(1-settings.decay, result.generationsTrained));

    switch (settings.algorithm) {
        case TrainingAlgorithm::GENETIC:
        case TrainingAlgorithm::NEUROEVOLUTION:
        case TrainingAlgorithm::RANDOM_SEARCH: {
            bool isGenetic = settings.algorithm == TrainingAlgorithm::GENETIC;
            std::cout << "Gen " << std::setw(5) << result.generationsTrained
              << " | Best" << std::setw(10) << std::fixed << std::setprecision(4) << result.bestFitness
              << " | Avg: " << std::setw(10) << std::setprecision(4) << result.averageFitness
              << " | " << (isGenetic ? "Mutation: " : "Noise Scale: ") << std::setw(10) << std::setprecision(4)
              << (isGenetic ? settings.mutationStdDev*decay : settings.noiseScale*decay)
              << std::endl;
            break;
        }
        case TrainingAlgorithm::GRADIENT_DESCENT: {
            std::cout << "Gen " << std::setw(5) << result.generationsTrained
              << " | RMSE" << std::setw(10) << std::fixed << std::setprecision(4) << result.bestRMSE
              << " | LR: " << std::setw(10) << std::setprecision(4)
              << settings.learningRate*decay << std::endl;
            break;
        }
    }
}

void NetworkTrainer::evaluatePopulationParallel(std::vector<NeuralNetwork>& population,
                                                std::vector<float>& fitness,
                                                const RewardFunction& reward,
                                                const TrainingSettings& settings,
                                                const uint32_t skipIdx = 0) {
    if (fitness.size() < population.size()) {
        fitness.resize(population.size());
    }

    uint32_t numThreads = settings.numThreads == 0 ? std::thread::hardware_concurrency() : settings.numThreads;
    uint32_t batchSize = settings.batchSize == 0 ? std::max(1u, uint32_t(population.size() / (numThreads * 4))) : settings.batchSize;

    std::mutex mtx;
    size_t nextIndex = skipIdx;

    auto worker = [&]() {
        while (true) {
            size_t startIdx, endIdx;
            {
                std::lock_guard lock(mtx);
                if (nextIndex >= population.size()) break;
                startIdx = nextIndex;
                endIdx = std::min(nextIndex + batchSize, population.size());
                nextIndex = endIdx;
            }

            std::vector<float>tempFit;
            for (size_t idx = startIdx; idx < endIdx; ++idx) {
                tempFit.push_back(evaluateNetwork(population[idx], reward));
            }

            {
                std::lock_guard lock(mtx);
                for (size_t idx = startIdx; idx < endIdx; ++idx) {
                    fitness[idx] = tempFit[idx-startIdx];
                }
            }
        }
    };

    std::vector<std::thread> threads;
    threads.reserve(numThreads);
    for (uint32_t i = 0; i < numThreads; ++i) {
        threads.emplace_back(worker);
    }

    for (auto& t : threads) {
        t.join();
    }
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
                std::cout << "Genetic\n";
                break;
            case TrainingAlgorithm::NEUROEVOLUTION:
                std::cout << "Neuroevolution\n";
                break;
            case TrainingAlgorithm::RANDOM_SEARCH:
                std::cout << "Random Search\n";
                break;
            case TrainingAlgorithm::GRADIENT_DESCENT:
                std::cout << "Gradient Descent\n";
                break;
        }
        std::cout << "Generations: " << settings.generations << std::endl;
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
        case TrainingAlgorithm::GRADIENT_DESCENT:
            trainGradientDescent(network, reward, settings);
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
        bool GD = settings.algorithm == TrainingAlgorithm::GRADIENT_DESCENT;

        logProgress(result, settings);
        std::cout << std::string(60, '-') << "\n";
        std::cout << "Training complete!\n";
        std::cout << "Best " << (GD ? "RMSE" : "fitness") << ": " << (GD ? result.bestRMSE : result.bestFitness) << "\n";
        if (!GD) std::cout << "Total evaluations: " << result.totalEvaluations << "\n";
        std::cout << "Total generations: " << result.generationsTrained << "\n";
        std::cout << "Training time: " << result.trainingTime.count() << " ms\n";
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
    fitness.resize(settings.populationSize);

    float mutationStdDev = settings.mutationStdDev;

    for (uint32_t i = 0; i < settings.populationSize; ++i) {
        population.push_back(network.clone());
        if (i > 0) {
            population[i].addNoise(mutationStdDev);
        }
    }

    // Evaluate initial population
    if (settings.enableMultithreading) {
        evaluatePopulationParallel(population, fitness, reward, settings);
    } else {
        for (size_t i = 0; i < settings.populationSize; ++i) {
            fitness[i] = evaluateNetwork(population[i], reward);
        }
    }

    // Main loop
    for (uint32_t gen = 0; gen < settings.generations; ++gen) {
        mutationStdDev *= 1-settings.decay;
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

        result.history.push_back(bestFit);
        result.averageFitness = avgFit;

        if (gen % settings.logInterval == 0) {
            logProgress(result, settings);
        }

        // Check termination conditions
        if (bestFit >= settings.targetFitness) {
            result.generationsTrained = gen + 1;
            return;
        }

        // Create next generation
        std::vector<NeuralNetwork> nextGen;
        std::vector<float> nextFitness;
        nextFitness.resize(settings.populationSize);

        // Elitism: keep top performers

        uint32_t idx = 0;
        uint32_t eliteSize = std::max(1u, uint32_t(float(settings.populationSize) * settings.elitePercent));
        for (uint32_t i = 0; i < eliteSize && i < indices.size(); ++i) {
            nextGen.push_back(population[indices[i]].clone());
            nextFitness[idx++] = fitness[indices[i]];
        }

        // Reproduction and mutation
        auto crossoverAmount = size_t(float(settings.populationSize) * settings.crossoverRate);
        std::uniform_int_distribution<size_t> parentSelection(0, std::min(crossoverAmount, indices.size() - 1));

        for (size_t n = eliteSize; n < settings.populationSize; ++n) {
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

            nextGen.push_back(std::move(offspring));
        }

        if (settings.enableMultithreading) {
            evaluatePopulationParallel(nextGen, nextFitness, reward, settings, eliteSize);
        } else {
            for (size_t i = eliteSize; i < settings.populationSize; ++i) {
                float fit = evaluateNetwork(nextGen[i], reward);
                nextFitness[i] = fit;
            }
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
    }

    if (settings.enableMultithreading) {
        evaluatePopulationParallel(population, fitness, reward, settings);
    } else {
        for (auto& individual : population) {
            fitness.push_back(evaluateNetwork(individual, reward));
        }
    }

    // Main loop
    for (uint32_t gen = 0; gen < settings.generations; ++gen) {
        noiseScale *= 1-settings.decay;
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

        result.history.push_back(bestFit);
        result.averageFitness = avgFit;

        if (gen % settings.logInterval == 0) {
            logProgress(result, settings);
        }

        // Check termination
        if (bestFit >= settings.targetFitness) {
            result.generationsTrained = gen + 1;
            return;
        }

        // Keep top specimens and mutate
        std::vector<NeuralNetwork> nextGen;
        std::vector<float> nextFitness;
        nextFitness.resize(settings.populationSize);

        uint32_t idx = 0;
        uint32_t topCount = std::min(settings.topSpecimens, uint32_t(indices.size()));
        for (uint32_t i = 0; i < topCount; ++i) {
            nextGen.push_back(population[indices[i]].clone());
            nextFitness[idx++] = fitness[indices[i]];
        }

        // Generate offspring from top specimens
        while (nextGen.size() < settings.populationSize) {
            size_t parentIdx = indices[rng() % topCount];
            auto offspring = population[parentIdx].clone();
            offspring.addNoise(noiseScale);

            nextGen.push_back(offspring);
        }

        if (settings.enableMultithreading) {
            std::vector offspringToEval(nextGen.begin() + topCount, nextGen.end());
            std::vector<float> offspringFitness;
            evaluatePopulationParallel(offspringToEval, offspringFitness, reward, settings);
            nextFitness.insert(nextFitness.end(), offspringFitness.begin(), offspringFitness.end());
        } else {
            for (uint32_t i = topCount; i < settings.populationSize; ++i) {
                float fit = evaluateNetwork(nextGen[i], reward);
                nextFitness[idx++] = fit;
            }
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
    while (iterations < settings.generations) {
        noiseScale *= 1-settings.decay;
        // Create random perturbation
        auto candidate = bestNetwork.clone();
        candidate.addNoise(noiseScale);

        float fit = evaluateNetwork(candidate, reward);

        if (fit > bestFit) {
            bestFit = fit;
            result.bestFitness = fit;
            bestNetwork = candidate.clone();
        }

        result.history.push_back(bestFit);

        if (iterations % settings.logInterval == 0) {
            logProgress(result, settings);
        }

        if (bestFit >= settings.targetFitness) {
            result.generationsTrained = iterations + 1;
            return;
        }

        iterations++;
    }

    result.generationsTrained = iterations;
}

void NetworkTrainer::trainGradientDescent(NeuralNetwork& network,
                                          const RewardFunction& reward,
                                          const TrainingSettings& settings) {
    if (!settings.trainingData) {
        throw std::invalid_argument("Training data required for gradient descent");
    }
    if (settings.trainingData->inputs.empty()) {
        throw std::invalid_argument("Training data is empty");
    }
    if (settings.learningRate <= 0) {
        throw std::invalid_argument("Learning rate must be positive");
    }

    auto startTime = std::chrono::high_resolution_clock::now();
    totalEvaluations = 0;
    result = TrainingResult();
    bestNetwork = network.clone();

    float learningRate = settings.learningRate;
    const auto& inputs = settings.trainingData->inputs;
    const auto& outputs = settings.trainingData->outputs;
    uint32_t dataSize = inputs.size();

    if (inputs.size() != outputs.size()) {
        throw std::invalid_argument("Input and output size mismatch");
    }

    if (settings.verbose) {
        std::cout << "Starting training with Gradient Descent (Backprop)\n"
                  << "Learning rate: " << learningRate << "\n"
                  << "Learning rate decay: " << settings.decay << "\n"
                  << "Batch size: " << (settings.batchSize == 0 ? dataSize : settings.batchSize) << "\n"
                  << std::string(60, '-') << std::endl;
    }

    std::mt19937 rng(settings.randomSeed ? settings.randomSeed : std::random_device{}());
    std::uniform_int_distribution<uint32_t> sampleDist(0, dataSize - 1);

    for (uint32_t gen = 0; gen < settings.generations; ++gen) {
        // Decay learning rate
        if (settings.decay > 0) {
            learningRate = settings.learningRate * std::pow(1.0f - settings.decay, float(gen));
        }

        // Determine batch size
        uint32_t batchSize = settings.batchSize == 0 ? dataSize : settings.batchSize;

        // Zero gradients before batch
        network.zeroGradients();

        // Process mini-batch
        float batchMSE = 0.0f;
        for (uint32_t b = 0; b < batchSize; ++b) {
            uint32_t idx = sampleDist(rng);

            // Forward pass
            auto predictions = network.forward(inputs[idx]);
            const auto& expected = outputs[idx];

            // Compute loss for all outputs
            float sampleMSE = 0.0f;
            std::vector<float> outputGradients(predictions.size());
            for (size_t i = 0; i < predictions.size(); ++i) {
                float error = predictions[i] - expected[i];
                sampleMSE += error * error;
                outputGradients[i] = 2.0f * error;  // MSE gradient
            }
            batchMSE += sampleMSE / float(predictions.size());

            // Backward pass
            network.backward(outputGradients);
        }

        batchMSE /= float(batchSize);

        // Update weights once per batch
        network.updateWeights(learningRate);

        // Evaluate on full dataset periodically
        if (gen % settings.logInterval == 0) {
            float fullDatasetMSE = 0.0f;
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto pred = network.forward(inputs[i]);
                for (size_t j = 0; j < pred.size(); ++j) {
                    float error = pred[j] - outputs[i][j];
                    fullDatasetMSE += error * error;
                }
            }
            fullDatasetMSE /= float(inputs.size() * network.getOutputSize());
            float rmse = std::sqrt(fullDatasetMSE);

            if (rmse < result.bestRMSE) {
                result.bestRMSE = rmse;
                bestNetwork = network.clone();
            }

            result.history.push_back(rmse);

            logProgress(result, settings);
        }

        // Check termination
        if (result.bestFitness >= settings.targetFitness) {
            result.generationsTrained = gen + 1;
            break;
        }

        result.generationsTrained = gen + 1;
    }

    auto endTime = std::chrono::high_resolution_clock::now();
    result.trainingTime = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
    network = bestNetwork.clone();
}