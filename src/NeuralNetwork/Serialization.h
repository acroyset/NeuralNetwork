//
// Created by Andreas Royset on 1/1/26.
//

#ifndef SERIALIZATION_H
#define SERIALIZATION_H



#pragma once

#include "NeuralNetwork.h"
#include <fstream>


class NetworkSerializer {
public:
	static void saveBinary(const NeuralNetwork& network, const std::string& filename);

	static NeuralNetwork loadBinary(const std::string& filename);

	static void saveJSON(const NeuralNetwork& network, const std::string& filename);

	static NeuralNetwork loadJSON(const std::string& filename);

private:
	static constexpr uint32_t BINARY_MAGIC = 0x4E4E4D4C; // "NNML"
	static constexpr uint32_t BINARY_VERSION = 1;

	static ActivationFunction* parseActivationFromName(const std::string& name);
};



#endif //SERIALIZATION_H
