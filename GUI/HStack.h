//
// Created by Andreas Royset on 1/23/26.
//

#ifndef HSTACK_H
#define HSTACK_H
#include "View.h"

class Spacer;

class HStack final : public View {
	std::vector<View*> items;

public:

	float spacing = 0;

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {

		int numItems = int(items.size());

		this->position = position;

		position.x += leftPadding;
		position.y += topPadding;

		float minHeightOfStack = 0.0f;
		float totalWidth = 0;
		int spacerCount = 0;

		// First pass: calculate non-spacer width and count spacers
		for (int i = 0; i < numItems; i++) {
			View* item = items[i];
			Spacer* spacer = dynamic_cast<Spacer*>(item);

			if (spacer) {
				spacerCount++;
			} else {
				float itemHeight = item->getHeight();
				float itemWidth = item->getWidth();
				totalWidth += itemWidth;

				if (itemHeight > minHeightOfStack) {
					minHeightOfStack = itemHeight;
				}
			}
		}

		// Calculate available space for spacers
		float availableWidth = width - leftPadding - rightPadding - totalWidth;
		float spacerWidth = spacerCount > 0 ? availableWidth / spacerCount : 0;

		// Second pass: update all items with proper positioning
		sf::Vector2f currentPos = position;
		for (int i = 0; i < numItems; i++) {
			View* item = items[i];
			Spacer* spacer = dynamic_cast<Spacer*>(item);

			if (spacer) {
				spacer->setPosition(currentPos);
				spacer->setWidth(spacerWidth);
				spacer->setHeight(minHeightOfStack);
				currentPos.x += spacerWidth;
			} else {
				item->update(currentPos, dt, mousePos, mousePressed);
				float itemHeight = item->getHeight();
				float itemWidth = item->getWidth();
				currentPos.x += itemWidth;
			}
		}

		width = totalWidth + leftPadding + rightPadding;
		height = minHeightOfStack + topPadding + bottomPadding;
	}

	void draw(sf::RenderTarget& target) const override {

		float rWidth = width - leftPadding - rightPadding;
		float rHeight = height - topPadding - bottomPadding;

		drawRoundedRectangle(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, bgColor, cornerRadius);
		drawRoundedOutline(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, outlineColor, outlineColor, cornerRadius, outlineThickness);

		for (const View* item : items) {
			item->draw(target);
		}
	}

	void addItem(View* item) {
		items.emplace_back(item);
	}
};

#endif //HSTACK_H