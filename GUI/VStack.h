//
// Created by Andreas Royset on 1/23/26.
//

#ifndef VSTACK_H
#define VSTACK_H
#include "View.h"
#include "Spacer.h"

class VStack final : public View {
	std::vector<View*> items;

public:

	float spacing = 0;

	void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {

		int numItems = int(items.size());

		this->position = position;

		position.x += leftPadding;
		position.y += topPadding;

		float minWidthOfStack = 0.0f;
		float totalHeight = 0;
		int spacerCount = 0;

		// First pass: calculate non-spacer height and count spacers
		for (int i = 0; i < numItems; i++) {
			View* item = items[i];
			Spacer* spacer = dynamic_cast<Spacer*>(item);

			if (spacer) {
				spacerCount++;
			} else {
				float itemHeight = item->getHeight();
				float itemWidth = item->getWidth();
				totalHeight += itemHeight;

				if (itemWidth > minWidthOfStack) {
					minWidthOfStack = itemWidth;
				}
			}
		}

		// Calculate available space for spacers
		float availableHeight = height - topPadding - bottomPadding - totalHeight;
		float spacerHeight = spacerCount > 0 ? availableHeight / spacerCount : 0;

		// Second pass: update all items with proper positioning
		sf::Vector2f currentPos = position;
		for (int i = 0; i < numItems; i++) {
			View* item = items[i];
			Spacer* spacer = dynamic_cast<Spacer*>(item);

			if (spacer) {
				spacer->setPosition(currentPos);
				spacer->setWidth(minWidthOfStack);
				spacer->setHeight(spacerHeight);
				currentPos.y += spacerHeight;
			} else {
				item->update(currentPos, dt, mousePos, mousePressed);
				float itemHeight = item->getHeight();
				currentPos.y += itemHeight;
			}
		}

		width = minWidthOfStack + leftPadding + rightPadding;
		height = totalHeight + topPadding + bottomPadding;
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

#endif //VSTACK_H