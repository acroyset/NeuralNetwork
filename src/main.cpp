#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Training.h"
#include "NeuralNetwork/Serialization.h"
#include <iostream>
#include <vector>


std::vector<std::vector<float>> inputs;
std::vector<std::vector<float>> outputs;

float evalFunction(const NeuralNetwork& network) {
    return 0;
}

int main() {
    for (int i = 0; i < 200; i++) {
        float x = float(rand()%100)/25-2;
        float y = float(rand()%100)/25-2;
        bool in = x+y <= 1;
        inputs.push_back({x, y});
        outputs.push_back({float(in), float(!in)});
    }

    std::vector<ActivationFunction*> activations = {
        new Tanh(),
        new Tanh(),
        new Sigmoid(),
    };

    // Larger hidden layers to memorize the mapping
    NeuralNetwork network(2, 2, {16, 16}, activations);

    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GRADIENT_DESCENT;
    settings.generations = 1000;
    settings.learningRate = 0.5f;
    settings.decayRate = 0.001f;
    settings.logInterval = 100;

    TrainingData data;
    data.inputs = outputs;
    data.outputs = inputs;
    settings.trainingData = &data;
    settings.targetPerformanceEnable = true;
    settings.targetPerformance = 0.0001;

    // Train
    NetworkTrainer trainer;
    auto result = trainer.train(network, evalFunction, settings);

    const auto& bestNet = trainer.getBestNetwork();

    std::vector<float> in = {0, 0.5};
    std::vector<float> out = bestNet.forward(in);
    std::cout << in[0] << " " << in[1] << " (" << sqrt(in[0]*in[0]+in[1]*in[1]) << ") " << out[0] << " " << out[1] << std::endl;


    NetworkSerializer::saveJSON(bestNet, "Models/model.json");

    return 0;
}