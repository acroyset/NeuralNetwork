//
// Created by Andreas Royset on 1/1/26.
//


#include "NeuralNetwork.h"
#include <algorithm>
#include <iostream>

// ==================== Layer Implementation ====================

Layer::Layer(size_t inputSize, size_t outputSize, ActivationFunction* activation)
    : activation(activation) {

    weights.resize(outputSize, std::vector<float>(inputSize));
    biases.resize(outputSize, 0.0f);
    deltas.resize(outputSize, 0.0f);

    // Initialize weights using Xavier initialization
    xavierInitialize();
}

void Layer::xavierInitialize() {
    std::mt19937 rng(std::random_device{}());
    size_t inputSize = weights[0].size();
    size_t outputSize = weights.size();

    float limit = std::sqrt(6.0f / float(inputSize + outputSize));
    std::uniform_real_distribution dist(-limit, limit);

    for (auto& row : weights) {
        for (auto& w : row) {
            w = dist(rng);
        }
    }

    std::fill(biases.begin(), biases.end(), 0.0f);
}

void Layer::heInitialize() {
    std::mt19937 rng(std::random_device{}());
    size_t inputSize = weights[0].size();

    float stdDev = std::sqrt(2.0f / float(inputSize));
    std::normal_distribution<float> dist(0.0f, stdDev);

    for (auto& row : weights) {
        for (auto& w : row) {
            w = dist(rng);
        }
    }

    std::fill(biases.begin(), biases.end(), 0.0f);
}

std::vector<float> Layer::forward(const std::vector<float>& input) const {
    if (input.size() != weights[0].size()) {
        throw std::invalid_argument("Input size mismatch");
    }

    inputs = input;
    outputs.resize(weights.size());

    for (size_t i = 0; i < weights.size(); ++i) {
        float sum = biases[i];
        for (size_t j = 0; j < input.size(); ++j) {
            sum += weights[i][j] * input[j];
        }
        outputs[i] = activation->activate(sum);
    }

    return outputs;
}

Layer Layer::clone() const {
    Layer newLayer(weights[0].size(), weights.size(), activation->clone());
    newLayer.weights = weights;
    newLayer.biases = biases;
    return newLayer;
}

// ==================== NeuralNetwork Implementation ====================

NeuralNetwork::NeuralNetwork(size_t inputSize,
                             size_t outputSize,
                             const std::vector<size_t>& hiddenSizes,
                             const std::vector<ActivationFunction*>& activations)
    : rng(std::random_device{}()) {

    if (activations.size() != hiddenSizes.size() + 1) {
        throw std::invalid_argument(
            "Number of activation functions must equal number of hidden layers + 1"
        );
    }

    // Create hidden layers
    size_t prevSize = inputSize;
    for (size_t i = 0; i < hiddenSizes.size(); ++i) {
        layers.emplace_back(prevSize, hiddenSizes[i], activations[i]->clone());
        prevSize = hiddenSizes[i];
    }

    // Create output layer
    layers.emplace_back(prevSize, outputSize, activations.back()->clone());
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& other)
    : rng(std::random_device{}()) {
    for (const auto& layer : other.layers) {
        layers.push_back(layer.clone());
    }
}

NeuralNetwork& NeuralNetwork::operator=(const NeuralNetwork& other) {
    if (this != &other) {
        layers.clear();
        for (const auto& layer : other.layers) {
            layers.push_back(layer.clone());
        }
    }
    return *this;
}

NeuralNetwork::NeuralNetwork(NeuralNetwork&& other) noexcept
    : layers(std::move(other.layers)), rng(std::random_device{}()) {}

std::vector<float> NeuralNetwork::forward(const std::vector<float>& input) const {
    std::vector<float> current = input;

    for (const auto& layer : layers) {
        current = layer.forward(current);
    }

    return current;
}

size_t NeuralNetwork::getInputSize() const {
    if (layers.empty()) return 0;
    return layers[0].getInputSize();
}

size_t NeuralNetwork::getOutputSize() const {
    if (layers.empty()) return 0;
    return layers.back().getOutputSize();
}

void NeuralNetwork::addNoise(float stdDev) {
    std::normal_distribution dist(0.0f, stdDev);

    for (auto& layer : layers) {
        auto& weights = layer.getWeights();
        for (auto& row : weights) {
            for (auto& w : row) {
                w += dist(rng);
            }
        }

        auto& biases = layer.getBiases();
        for (auto& b : biases) {
            b += dist(rng);
        }
    }
}

NeuralNetwork NeuralNetwork::clone() const {
    NeuralNetwork copy(*this);
    return copy;
}