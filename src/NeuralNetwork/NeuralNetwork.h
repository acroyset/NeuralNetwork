//
// Created by Andreas Royset on 1/1/26.
//

#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H



#pragma once

#include <vector>
#include <random>
#include <memory>
#include <stdexcept>
#include "activationFunctions.h"


class Layer {
    std::vector<std::vector<float>> weights;  // [output_size][input_size]
    std::vector<float> biases;                 // [output_size]
    std::unique_ptr<ActivationFunction> activation;

    // For backprop
    mutable std::vector<float> inputs;
    mutable std::vector<float> outputs;
    mutable std::vector<float> deltas;

public:
    Layer(size_t inputSize, size_t outputSize, ActivationFunction* activation);

    std::vector<float> forward(const std::vector<float>& input) const;

    std::vector<std::vector<float>>& getWeights() { return weights; }  // MUTABLE
    const std::vector<std::vector<float>>& getWeights() const { return weights; }  // CONST
    void setWeights(const std::vector<std::vector<float>>& w) { weights = w; }

    std::vector<float>& getBiases() { return biases; }  // MUTABLE
    const std::vector<float>& getBiases() const { return biases; }  // CONST
    void setBiases(const std::vector<float>& b) { biases = b; }

    void xavierInitialize();

    void heInitialize();

    size_t getOutputSize() const { return biases.size(); }

    size_t getInputSize() const {
        return weights.empty() ? 0 : weights[0].size();
    }

    std::string getActivationName() const {
        return activation->getName();
    }

    const std::vector<float>& getOutputs() const { return outputs; }

    const std::vector<float>& getInputs() const { return inputs; }

    Layer clone() const;
};

class NeuralNetwork {
    std::vector<Layer> layers;
    std::mt19937 rng;

public:
    NeuralNetwork(size_t inputSize,
                  size_t outputSize,
                  const std::vector<size_t>& hiddenSizes,
                  const std::vector<ActivationFunction*>& activations);

    NeuralNetwork(const NeuralNetwork& other);

    NeuralNetwork& operator=(const NeuralNetwork& other);

    NeuralNetwork(NeuralNetwork&& other) noexcept;

    [[nodiscard]] std::vector<float> forward(const std::vector<float>& input) const;

    [[nodiscard]] size_t getLayerCount() const { return layers.size(); }

    Layer& getLayer(size_t index) {
        if (index >= layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers[index];
    }

    const Layer& getLayer(size_t index) const {
        if (index >= layers.size()) {
            throw std::out_of_range("Layer index out of range");
        }
        return layers[index];
    }

    size_t getInputSize() const;

    size_t getOutputSize() const;

    std::vector<Layer>& getLayers() { return layers; }

    const std::vector<Layer>& getLayers() const { return layers; }

    void addNoise(float stdDev);

    NeuralNetwork clone() const;
};



#endif //NEURALNETWORK_H
