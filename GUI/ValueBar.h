//
// Created by Andreas Royset on 1/24/26.
//

#ifndef VALUEBAR_H
#define VALUEBAR_H

#include <iomanip>
#include <sstream>

#include "Utils.h"
#include "Text.h"
#include "View.h"

class ValueBar final : public View {

	Text* textObj;

public:

	const float& barValue;
	float minValue;
	float maxValue;
	float barWidth;
	float barHeight;

	float padding = 16;
	float outlineThickness = 4;

	ValueBar(const float& barValue, float minValue, float maxValue, float barWidth, float barHeight, Text* text) :
			barValue(barValue),
			minValue(minValue),
			maxValue(maxValue),
			barWidth(barWidth),
			barHeight(barHeight),
			textObj(text) {}

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {

		textObj->update(position + sf::Vector2f(leftPadding, topPadding), dt, mousePos, mousePressed);

		this->position = position;

		width = fmaxf(textObj->getWidth(), barWidth+2*padding) + leftPadding + rightPadding;
		height = textObj->getHeight() + barHeight + topPadding + bottomPadding + 2*padding;
	}

	void draw(sf::RenderTarget& target) const override {

		float rWidth = width - leftPadding - rightPadding;
		float rHeight = height - topPadding - bottomPadding;

		drawRoundedRectangle(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, bgColor, cornerRadius);
		drawRoundedOutline(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, outlineColor, outlineColor, cornerRadius, outlineThickness);

		sf::Vector2f startPos = position + sf::Vector2f{leftPadding, topPadding} + sf::Vector2f{padding, textObj->getHeight() + padding};
		sf::Vector2f endtPos = startPos + sf::Vector2f{barWidth-2*padding, 0};

		float scale = (barValue - minValue) / (maxValue - minValue);
		scale = fminf(fmaxf(scale, 0), 1);

		drawRoundedOutline(target, startPos, barWidth, barHeight, sf::Color(130, 130, 130), sf::Color(130, 130, 130), cornerRadius-padding, outlineThickness);
		drawRoundedRectangle(target, startPos+sf::Vector2f(outlineThickness, outlineThickness), (barWidth-2*outlineThickness) * scale, barHeight-2*outlineThickness, textObj->color, cornerRadius-padding-outlineThickness);

		int decimalPlaces = 2;

		for (int i = 0; i < 4; i++) {

			float t = float(i) / 3;

			std::ostringstream oss;
			oss << std::fixed << std::setprecision(decimalPlaces) << minValue * (1-t) + maxValue * t;
			sf::Text label(oss.str(), textObj->font, 12);
			label.setFillColor(sf::Color(130, 130, 130));
			label.setPosition(startPos * (1-t) + endtPos * t + sf::Vector2f(0, -15));
			target.draw(label);
		}

		textObj->draw(target);
	}
};

#endif //VALUEBAR_H
