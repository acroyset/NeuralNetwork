#include "NeuralNetwork/NeuralNetwork.h"
#include "NeuralNetwork/Training.h"
#include "NeuralNetwork/Serialization.h"
#include <iostream>
#include <thread>
#include <vector>

#include "../GUI/Graph.h"
#include "../GUI/Text.h"
#include "../GUI/VStack.h"
#include "../GUI/HStack.h"
#include "../GUI/Root.h"
#include "../GUI/ValueBar.h"


std::vector<std::vector<float>> inputs;
std::vector<std::vector<float>> outputs;

float evalFunction(const NeuralNetwork& network) {
    return 0;
}


int main() {
    for (int i = 0; i < 20; i++) {
        float x = float(i)/20.0f;
        float out = x;
        inputs.push_back({x});
        outputs.push_back({out});
    }

    std::vector<ActivationFunction*> activations = {
        new Linear()
    };

    // Larger hidden layers to memorize the mapping
    NeuralNetwork network(1, 1, {}, activations);

    TrainingSettings settings;
    settings.algorithm = TrainingAlgorithm::GRADIENT_DESCENT;
    settings.generations = 10000;
    settings.learningRate = 0.5f;
    settings.decayRate = 0.000f;
    settings.logInterval = 100;
    settings.verbose = true;

    TrainingData data;
    data.inputs = inputs;
    data.outputs = outputs;
    settings.trainingData = &data;
    settings.targetPerformanceEnable = true;
    settings.targetPerformance = 0.0001;

    // Train
    NetworkTrainer trainer;

    std::thread trainerThread([&network, settings, &trainer] {
        trainer.train(network, evalFunction, settings);
    });

    sf::Font font;
    if (!font.loadFromFile("/System/Library/Fonts/SFNSRounded.ttf")) {
        std::cerr << "Error loading font\n";
        return -1;
    }

    float generation = 0;

    auto* textObj1 = new Text("Hello World", Color::White, font, 30);
    textObj1->setBgColor(Color::Black);
    textObj1->setPadding(8);
    textObj1->setCornerRadius(32);
    textObj1->setTextPadding(32);
    textObj1->setOutlineColor(Color::White);
    textObj1->setOutlineThickness();

    std::string text2 = "FPS: ";
    auto* textObj2 = new Text(text2, Color::White, font, 30);
    textObj2->setBgColor(Color::Black);
    textObj2->setPadding(8);
    textObj2->setCornerRadius(32);
    textObj2->setTextPadding(32);
    textObj2->setOutlineColor(Color::White);
    textObj2->setOutlineThickness();

    auto* blankTextObj = new Text("", Color::White, font, 30);

    auto* valueBar1 = new ValueBar(generation, 0, settings.generations, 200, 32, blankTextObj);
    valueBar1->setBgColor(Color::Red);
    valueBar1->setPadding(8);
    valueBar1->setCornerRadius(32);

    std::string graph1Title = "Graph";
    auto* graphTitle = new Text(graph1Title, Color::White, font, 30);
    graphTitle->setTextPadding(32);

    std::vector<sf::Vector2f> gdata {};
    auto* graph1 = new Graph(gdata, 500, 500, graphTitle);
    graph1->setBgColor(Color::Yellow);
    graph1->setPadding(8);
    graph1->setCornerRadius(32);

    auto* VStack1 = new VStack();
    VStack1->addItem(textObj1);
    VStack1->addItem(textObj2);
    VStack1->addItem(valueBar1);

    auto* HStack1 = new HStack();
    HStack1->addItem(VStack1);
    HStack1->addItem(new Spacer());
    HStack1->addItem(graph1);
    HStack1->setPadding(8);

    auto* root = new Root();

    root->addView(HStack1);

    float x = 0;

    sf::Clock clock;
    while (root->isOpen()) {

        float dt = clock.restart().asSeconds();
        text2 = "Fps: " + std::to_string(int(1/dt));
        TrainingResult result = trainer.getResult();
        generation = float(result.generationsTrained);

        gdata.emplace_back(generation, result.bestRMSE);
        x += dt;

        root->update(dt);
        root->draw();
    }

    delete root;


    trainerThread.join();

    const auto& bestNet = trainer.getBestNetwork();

    std::vector<float> in = {0, 0.5};
    std::vector<float> out = bestNet.forward(in);
    std::cout << in[0] << " " << in[1] << " (" << sqrt(in[0]*in[0]+in[1]*in[1]) << ") " << out[0] << " " << out[1] << std::endl;


    NetworkSerializer::saveJSON(bestNet, "Models/model.json");

    return 0;
}