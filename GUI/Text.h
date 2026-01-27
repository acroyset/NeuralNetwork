//
// Created by Andreas Royset on 1/23/26.
//

#ifndef TEXT_H
#define TEXT_H

#include <variant>
#include "Utils.h"
#include "View.h"

class Text final : public View {

	sf::Text textObj;
	std::variant<std::string, std::string*> textVariant;
	bool isReference;

public:

	sf::Color color;
	sf::Font font;

	float reserveWidth = 0;

	int charSize;

	float textPadding = 16;

    // Constructor for constant string
    Text(const std::string& text, sf::Color color, const sf::Font& font, int charSize)
        : textVariant(std::string(text)), isReference(false), color(color), font(font), charSize(charSize) {
       textObj = sf::Text(std::get<std::string>(textVariant), font, charSize);
       textObj.setFillColor(color);
       textObj.setStyle(sf::Text::Bold);
    }

    // Constructor for reference to string
    Text(std::string& text, sf::Color color, const sf::Font& font, int charSize)
        : textVariant(&text), isReference(true), color(color), font(font), charSize(charSize) {
		textObj = sf::Text(text, font, charSize);
		textObj.setFillColor(color);
		textObj.setStyle(sf::Text::Bold);
	}

    std::string& getText() {
        if (isReference) {
            return *std::get<std::string*>(textVariant);
        } else {
            return std::get<std::string>(textVariant);
        }
    }

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {

		textObj.setString(getText());

    	this->position = position;
    	textObj.setPosition(position + sf::Vector2f(textPadding + leftPadding, textPadding + topPadding - charSize/4));

    	if (getText() == "") {
    		width = 0;
    		height = 0;
    		return;
    	}

		sf::FloatRect bounds = textObj.getGlobalBounds();
    	bounds.height = charSize * 7/7;

		width = bounds.width + 2*textPadding + leftPadding + rightPadding;
		height = bounds.height + 2*textPadding + topPadding + bottomPadding;

		width = fmaxf(width, reserveWidth);
		if (width > reserveWidth) reserveWidth= width + 20;
	}

	void draw(sf::RenderTarget& target) const override {

		float rWidth = width - leftPadding - rightPadding;
		float rHeight = height - topPadding - bottomPadding;

		drawRoundedRectangle(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, bgColor, cornerRadius);
		drawRoundedOutline(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, outlineColor, outlineColor, cornerRadius, outlineThickness);

		target.draw(textObj);
	}

	void setTextPadding(float textPadding) {this->textPadding = textPadding;}
	void setReservedWidth(float reservedWidth) {this->reserveWidth = reservedWidth;}
};

#endif //TEXT_H