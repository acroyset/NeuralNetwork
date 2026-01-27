//
// Created by Andreas Royset on 1/23/26.
//

#ifndef SPACER_H
#define SPACER_H
#include "View.h"

class Spacer final : public View {

public:

	bool isExpanding = true;

	Spacer() {}

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {
		this->position = position;
		width = 0;
		height = 0;
	}

	void draw(sf::RenderTarget& target) const override {
		// Spacer draws nothing
	}
};

#endif //SPACER_H