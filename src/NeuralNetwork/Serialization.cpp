//
// Created by Andreas Royset on 1/1/26.
//

#include "serialization.h"
#include <iostream>


ActivationFunction* NetworkSerializer::parseActivationFromName(const std::string& name) {
    if (name == "ReLU") return new ReLU();
    if (name == "Sigmoid") return new Sigmoid();
    if (name == "Tanh") return new Tanh();
    if (name == "Linear") return new Linear();
    if (name.find("LeakyReLU") == 0) return new LeakyReLU(0.01f);
    if (name.find("ELU") == 0) return new ELU(1.0f);
    if (name == "SELU") return new SELU();

    throw std::runtime_error("Unknown activation function: " + name);
}

void NetworkSerializer::saveBinary(const NeuralNetwork& network, const std::string& filename) {
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    // Write header
    file.write(reinterpret_cast<const char*>(&BINARY_MAGIC), sizeof(BINARY_MAGIC));
    file.write(reinterpret_cast<const char*>(&BINARY_VERSION), sizeof(BINARY_VERSION));

    // Write network structure
    uint32_t numLayers = network.getLayerCount();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(numLayers));

    // Write each layer
    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        const auto& layer = network.getLayer(i);

        uint32_t inputSize = layer.getInputSize();
        uint32_t outputSize = layer.getOutputSize();

        file.write(reinterpret_cast<const char*>(&inputSize), sizeof(inputSize));
        file.write(reinterpret_cast<const char*>(&outputSize), sizeof(outputSize));

        // Write activation function name
        std::string activationName = layer.getActivationName();
        uint32_t nameLen = activationName.length();
        file.write(reinterpret_cast<const char*>(&nameLen), sizeof(nameLen));
        file.write(activationName.c_str(), nameLen);

        // Write weights
        const auto& weights = layer.getWeights();
        for (const auto& row : weights) {
            for (float w : row) {
                file.write(reinterpret_cast<const char*>(&w), sizeof(float));
            }
        }

        // Write biases
        const auto& biases = layer.getBiases();
        for (float b : biases) {
            file.write(reinterpret_cast<const char*>(&b), sizeof(float));
        }
    }

    file.close();
    std::cout << "Network saved to: " << filename << std::endl;
}

NeuralNetwork NetworkSerializer::loadBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    // Read header
    uint32_t magic, version;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    file.read(reinterpret_cast<char*>(&version), sizeof(version));

    if (magic != BINARY_MAGIC) {
        throw std::runtime_error("Invalid file format (magic number mismatch)");
    }

    if (version != BINARY_VERSION) {
        throw std::runtime_error("Incompatible file version");
    }

    // Read network structure
    uint32_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(numLayers));

    std::vector<size_t> hiddenSizes;
    std::vector<ActivationFunction*> activations;
    size_t inputSize = 0;
    size_t outputSize = 0;

    // First pass: read layer info to reconstruct architecture
    std::vector<std::tuple<uint32_t, uint32_t, std::string>> layerInfo;
    std::streampos layerDataStart = file.tellg();

    for (uint32_t i = 0; i < numLayers; ++i) {
        uint32_t inSize, outSize;
        file.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
        file.read(reinterpret_cast<char*>(&outSize), sizeof(outSize));

        uint32_t nameLen;
        file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        std::string activationName(nameLen, '\0');
        file.read(&activationName[0], nameLen);

        layerInfo.emplace_back(inSize, outSize, activationName);

        if (i == 0) inputSize = inSize;
        outputSize = outSize;

        if (i < numLayers - 1) {
            hiddenSizes.push_back(outSize);
        }

        activations.push_back(parseActivationFromName(activationName));

        // Skip weights and biases for now
        file.seekg(static_cast<long>(inSize * outSize * sizeof(float) +
                                     outSize * sizeof(float)),
                   std::ios::cur);
    }

    // Create the network
    NeuralNetwork network(inputSize, outputSize, hiddenSizes, activations);

    // Second pass: load weights and biases
    file.seekg(layerDataStart);

    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        auto& layer = network.getLayer(i);
        uint32_t inSize, outSize;

        file.read(reinterpret_cast<char*>(&inSize), sizeof(inSize));
        file.read(reinterpret_cast<char*>(&outSize), sizeof(outSize));

        uint32_t nameLen;
        file.read(reinterpret_cast<char*>(&nameLen), sizeof(nameLen));
        file.seekg(nameLen, std::ios::cur); // Skip activation name

        // Read weights
        std::vector<std::vector<float>> weights(outSize, std::vector<float>(inSize));
        for (uint32_t j = 0; j < outSize; ++j) {
            for (uint32_t k = 0; k < inSize; ++k) {
                file.read(reinterpret_cast<char*>(&weights[j][k]), sizeof(float));
            }
        }
        layer.setWeights(weights);

        // Read biases
        std::vector<float> biases(outSize);
        for (uint32_t j = 0; j < outSize; ++j) {
            file.read(reinterpret_cast<char*>(&biases[j]), sizeof(float));
        }
        layer.setBiases(biases);
    }

    file.close();
    std::cout << "Network loaded from: " << filename << std::endl;
    return network;
}

void NetworkSerializer::saveJSON(const NeuralNetwork& network, const std::string& filename) {
    std::filesystem::path filepath(filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for writing: " + filename);
    }

    file << "{\n";
    file << "  \"version\": 1,\n";
    file << "  \"input_size\": " << network.getInputSize() << ",\n";
    file << "  \"output_size\": " << network.getOutputSize() << ",\n";
    file << "  \"layers\": [\n";

    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        const auto& layer = network.getLayer(i);

        file << "    {\n";
        file << "      \"index\": " << i << ",\n";
        file << "      \"input_size\": " << layer.getInputSize() << ",\n";
        file << "      \"output_size\": " << layer.getOutputSize() << ",\n";
        file << R"(      "activation": ")" << layer.getActivationName() << "\",\n";

        // Write weights
        file << "      \"weights\": [\n";
        const auto& weights = layer.getWeights();
        for (size_t j = 0; j < weights.size(); ++j) {
            file << "        [";
            for (size_t k = 0; k < weights[j].size(); ++k) {
                file << std::scientific << std::setprecision(8) << weights[j][k];
                if (k < weights[j].size() - 1) file << ", ";
            }
            file << "]";
            if (j < weights.size() - 1) file << ",";
            file << "\n";
        }
        file << "      ],\n";

        // Write biases
        file << "      \"biases\": [";
        const auto& biases = layer.getBiases();
        for (size_t j = 0; j < biases.size(); ++j) {
            file << std::scientific << std::setprecision(8) << biases[j];
            if (j < biases.size() - 1) file << ", ";
        }
        file << "]\n";

        file << "    }";
        if (i < network.getLayerCount() - 1) file << ",";
        file << "\n";
    }

    file << "  ]\n";
    file << "}\n";

    file.close();
    std::cout << "Network saved to JSON: " << filename << std::endl;
}

NeuralNetwork NetworkSerializer::loadJSON(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file for reading: " + filename);
    }

    std::string content((std::istreambuf_iterator<char>(file)),
                        std::istreambuf_iterator<char>());
    file.close();

    // Simple JSON parsing (manual, no library needed)
    size_t inputSizeStart = content.find("\"input_size\": ") + 14;
    size_t inputSizeEnd = content.find(',', inputSizeStart);
    size_t inputSize = std::stoi(content.substr(inputSizeStart, inputSizeEnd - inputSizeStart));

    size_t outputSizeStart = content.find("\"output_size\": ") + 15;
    size_t outputSizeEnd = content.find(',', outputSizeStart);
    size_t outputSize = std::stoi(content.substr(outputSizeStart, outputSizeEnd - outputSizeStart));

    // Extract layer information
    std::vector<size_t> hiddenSizes;
    std::vector<ActivationFunction*> activations;

    size_t layersStart = content.find("\"layers\": [");
    size_t layersEnd = content.rfind(']');
    std::string layersStr = content.substr(layersStart, layersEnd - layersStart);

    // Find each layer
    size_t layerPos = 0;
    while ((layerPos = layersStr.find("\"index\":", layerPos)) != std::string::npos) {
        // Get layer output size
        size_t outSizeStart = layersStr.find("\"output_size\": ", layerPos) + 15;
        size_t outSizeEnd = layersStr.find(',', outSizeStart);
        size_t layerOutSize = std::stoi(layersStr.substr(outSizeStart, outSizeEnd - outSizeStart));

        // Get activation function
        size_t actStart = layersStr.find(R"("activation": ")", layerPos) + 15;
        size_t actEnd = layersStr.find('\"', actStart);
        std::string activationName = layersStr.substr(actStart, actEnd - actStart);

        activations.push_back(parseActivationFromName(activationName));

        // Add to hidden sizes if not last layer
        layerPos = outSizeEnd;

        // Peek ahead to see if this is the output layer
        size_t nextLayerPos = layersStr.find("\"index\":", layerPos);
        if (nextLayerPos != std::string::npos) {
            // Not last layer
            hiddenSizes.push_back(layerOutSize);
        }
    }

    // Create network with architecture
    NeuralNetwork network(inputSize, outputSize, hiddenSizes, activations);

    // Now load weights and biases
    layerPos = 0;
    for (size_t i = 0; i < network.getLayerCount(); ++i) {
        // Find this layer's weights
        layerPos = layersStr.find("\"index\": " + std::to_string(i), layerPos);

        size_t weightsStart = layersStr.find("\"weights\": [", layerPos) + 12;
        size_t weightsEnd = layersStr.find(']', weightsStart) + 1;
        std::string weightsStr = layersStr.substr(weightsStart, weightsEnd - weightsStart);

        // Parse weights matrix
        std::vector<std::vector<float>> weights;
        size_t rowStart = 0;
        while ((rowStart = weightsStr.find('[', rowStart)) != std::string::npos) {
            size_t rowEnd = weightsStr.find(']', rowStart);
            std::string rowStr = weightsStr.substr(rowStart + 1, rowEnd - rowStart - 1);

            std::vector<float> row;
            size_t numStart = 0;
            while ((numStart = rowStr.find_first_not_of(" ,", numStart)) != std::string::npos) {
                size_t numEnd = rowStr.find_first_of(" ,", numStart);
                if (numEnd == std::string::npos) numEnd = rowStr.length();

                float value = std::stof(rowStr.substr(numStart, numEnd - numStart));
                row.push_back(value);
                numStart = numEnd;
            }

            if (!row.empty()) {
                weights.push_back(row);
            }
            rowStart = rowEnd + 1;
        }

        if (!weights.empty()) {
            network.getLayer(i).setWeights(weights);
        }

        // Find this layer's biases
        size_t biasesStart = layersStr.find("\"biases\": [", layerPos) + 12;
        size_t biasesEnd = layersStr.find(']', biasesStart);
        std::string biasesStr = layersStr.substr(biasesStart, biasesEnd - biasesStart);

        // Parse biases vector
        std::vector<float> biases;
        size_t numStart = 0;
        while ((numStart = biasesStr.find_first_not_of(" ,", numStart)) != std::string::npos) {
            size_t numEnd = biasesStr.find_first_of(" ,", numStart);
            if (numEnd == std::string::npos) numEnd = biasesStr.length();

            float value = std::stof(biasesStr.substr(numStart, numEnd - numStart));
            biases.push_back(value);
            numStart = numEnd;
        }

        if (!biases.empty()) {
            network.getLayer(i).setBiases(biases);
        }
    }

    std::cout << "Network loaded from JSON: " << filename << std::endl;
    return network;
}
