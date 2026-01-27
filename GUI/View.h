//
// Created by Andreas Royset on 1/23/26.
//

#ifndef VIEW_H
#define VIEW_H

#include <SFML/Graphics.hpp>

#include "Interpolated.h"

struct View {

protected:

	sf::Vector2f position;

	float topPadding = 0;
	float bottomPadding = 0;
	float rightPadding = 0;
	float leftPadding = 0;

	float width = 0;
	float height = 0;

	sf::Color bgColor = sf::Color::Transparent;
	sf::Color outlineColor = sf::Color::Transparent;
	float outlineThickness = 0;

	float cornerRadius = 0.1;

public:
	virtual ~View() = default;

	View() = default;

	virtual void update(sf::Vector2f, float, sf::Vector2f, bool) = 0;
	virtual void draw(sf::RenderTarget&) const = 0;

	void setTopPadding(float topPadding = 16) { this->topPadding = topPadding; }
	void setBottomPadding(float bottomPadding = 16) { this->bottomPadding = bottomPadding; }
	void setLeftPadding(float leftPadding = 16) { this->leftPadding = leftPadding; }
	void setRightPadding(float rightPadding = 16) { this->rightPadding = rightPadding; }

	void setVerticalPadding(float verticalPadding = 16) {
		this->topPadding = verticalPadding;
		this->bottomPadding = verticalPadding;
	}
	void setHorizontalPadding(float horizontalPadding = 16) {
		this->leftPadding = horizontalPadding;
		this->rightPadding = horizontalPadding;
	}
	void setPadding(float padding = 16) {
		this->topPadding = padding;
		this->bottomPadding = padding;
		this->leftPadding = padding;
		this->rightPadding = padding;
	}

	void setCornerRadius(float cornerRadius = 8) {this->cornerRadius = cornerRadius;}

	void setBgColor(sf::Color color) {this->bgColor = color;}
	void setOutlineColor(sf::Color color) {this->outlineColor = color; outlineThickness = fmaxf(outlineThickness, 1.0);}
	void setOutlineThickness(float outlineThickness = 4) {this->outlineThickness = outlineThickness;}

	void setPosition(sf::Vector2f pos) { this->position = pos; }
	void setWidth(float w) { this->width = w; }
	void setHeight(float h) { this->height = h; }

	[[nodiscard]] float getWidth() const {return width;}
	[[nodiscard]] float getHeight() const {return height;}
	[[nodiscard]] sf::Vector2f getPosition() const {return position;}
};

#endif //VIEW_H