//
// Created by Andreas Royset on 1/23/26.
//

#ifndef GRAPH_H
#define GRAPH_H
#include <iomanip>
#include <sstream>

#include "Utils.h"
#include "Text.h"
#include "View.h"

class Graph final : public View {

	Text* textObj;

public:

	std::vector<sf::Vector2f>& points;

	float graphWidth;
	float graphHeight;

	float padding = 16;
	float pointRadius = 1;
	float lineThickness = 3;
	float maxPoints = 256;

	Graph(std::vector<sf::Vector2f>& data, float graphWidth, float graphHeight, Text* text) :
			points(data),
			graphWidth(graphWidth),
			graphHeight(graphHeight),
			textObj(text) {}

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {

		textObj->update(position + sf::Vector2f(leftPadding, topPadding), dt, mousePos, mousePressed);

		this->position = position;

		float textHeight = textObj->getHeight();

		width = fmaxf(textObj->getWidth(), graphWidth) + leftPadding + rightPadding;
		height = textHeight + graphHeight + topPadding + bottomPadding + (textHeight != 0 ? padding : 0);
	}

	void draw(sf::RenderTarget& target) const override {

		float rWidth = width - leftPadding - rightPadding;
		float rHeight = height - topPadding - bottomPadding;

		drawRoundedRectangle(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, bgColor, cornerRadius);
		drawRoundedOutline(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, outlineColor, outlineColor, cornerRadius, outlineThickness);

		sf::Vector2f startPos = position + sf::Vector2f{leftPadding + padding,topPadding + textObj->getHeight()};
		float GWidth = rWidth-2*padding;
		float GHeight = rHeight - textObj->getHeight() - padding;

		drawRoundedRectangle(target, startPos, GWidth, GHeight, sf::Color(57, 57, 57), cornerRadius - padding);

		sf::Vector2f mins{static_cast<float>(pow(10, 307)), static_cast<float>(pow(10,307))};
		sf::Vector2f maxs{static_cast<float>(-pow(10, 307)), static_cast<float>(-pow(10,307))};

		for (sf::Vector2f& point : points) {
			mins.x = std::min(mins.x, point.x);
			mins.y = std::min(mins.y, point.y);
			maxs.x = std::max(maxs.x, point.x);
			maxs.y = std::max(maxs.y, point.y);
		}

		float graphPadding = cornerRadius - padding;

		float rangeX = maxs.x - mins.x;
		float rangeY = maxs.y - mins.y;

		if (rangeX == 0) rangeX = 1;
		if (rangeY == 0) rangeY = 1;

		float scaleX = (GWidth-2*pointRadius - 2*graphPadding) / rangeX;
		float scaleY = (GHeight-2*pointRadius - 2*graphPadding) / rangeY;

		// Draw gridlines at fixed data intervals that scale with mins/maxs
		float gridStepX = std::pow(2, std::floor(std::log2(rangeX / 4)));
		float gridStepY = std::pow(2, std::floor(std::log2(rangeY / 4)));

		int decimalPlaces = 2;

		int labelCountX = 0;
		int labelCountY = 0;

		// Vertical gridlines with labels
		for (float dataX = std::ceil(mins.x / gridStepX) * gridStepX; dataX <= maxs.x; dataX += gridStepX) {
			float screenX = (dataX - mins.x) * scaleX + startPos.x + graphPadding;

			drawThickLine(
				target,
				sf::Vector2f{screenX, startPos.y + graphPadding},
				sf::Vector2f{screenX, startPos.y + GHeight - graphPadding + (labelCountX % 2 == 0 ? 3 : 0)},
				2,
				sf::Color(80, 80, 80)
			);

			if (labelCountX % 2 == 0) {
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(decimalPlaces) << dataX;
				sf::Text label(oss.str(), textObj->font, 12);
				label.setFillColor(sf::Color(150, 150, 150));
				label.setPosition(screenX - 10, startPos.y + GHeight - graphPadding);
				target.draw(label);
			}
			labelCountX++;
		}

		// Horizontal gridlines with labels
		for (float dataY = std::ceil(mins.y / gridStepY) * gridStepY; dataY <= maxs.y; dataY += gridStepY) {
			float screenY = startPos.y + GHeight - graphPadding - (dataY - mins.y) * scaleY;

			drawThickLine(
				target,
				sf::Vector2f{startPos.x + graphPadding, screenY},
				sf::Vector2f{startPos.x + GWidth - graphPadding, screenY},
				2,
				sf::Color(80, 80, 80)
			);

			if (labelCountY % 2 == 0 && dataY != 0) {
				std::ostringstream oss;
				oss << std::fixed << std::setprecision(decimalPlaces) << dataY;
				sf::Text label(oss.str(), textObj->font, 12);
				label.setFillColor(sf::Color(150, 150, 150));
				label.setPosition(startPos.x + GWidth - label.getLocalBounds().width - graphPadding, screenY);
				target.draw(label);
			}
			labelCountY++;
		}

		int step = std::max(1.0f, float(points.size()/maxPoints));

		for (int i = 0; i < points.size()-1; i += step) {
			i = std::min(i, int(points.size())-1-step);
			sf::Vector2f point0 = points[i];
			sf::Vector2f normalizedPoint0 = {
				(point0.x - mins.x) * scaleX  +  startPos.x + pointRadius + graphPadding,
				(maxs.y - point0.y) * scaleY  +  startPos.y + pointRadius + graphPadding
			};

			sf::Vector2f point1 = points[i+step];
			sf::Vector2f normalizedPoint1 = {
				(point1.x - mins.x) * scaleX  +  startPos.x + pointRadius + graphPadding,
				(maxs.y - point1.y) * scaleY  +  startPos.y + pointRadius + graphPadding
			};

			drawThickLine(target, normalizedPoint0, normalizedPoint1, lineThickness, bgColor);
		}

		sf::CircleShape circle(pointRadius);
		for (int i = 0; i < points.size()+step-1; i += step) {
			i = std::min(i, int(points.size())-1);
			sf::Vector2f point = points[i];

			sf::Vector2f normalizedPoint = {
				(point.x - mins.x) * scaleX  +  startPos.x + graphPadding,
				(maxs.y - point.y) * scaleY  +  startPos.y + graphPadding
			};

			circle.setPosition(normalizedPoint);
			circle.setFillColor(bgColor);

			target.draw(circle);

		}

		textObj->draw(target);
	}
};

#endif //GRAPH_H
