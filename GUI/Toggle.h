//
// Created by Andreas Royset on 1/24/26.
//

#ifndef TOGGLE_H
#define TOGGLE_H


#include <iomanip>
#include <sstream>

#include "Utils.h"
#include "Text.h"
#include "View.h"

class Toggle final : public View {

    Text* textObj;

    bool state;

    Interpolated<float> togglePosition;

    float clickCooldown;

public:

    float toggleWidth;
    float toggleHeight;

    float padding = 8;
    float outlineThickness = 4;

    Toggle(bool state, float barWidth, float barHeight, Text* text) :
          state(state),
          toggleWidth(barWidth),
          toggleHeight(barHeight),
          textObj(text),
          togglePosition(state, 0.25),
          clickCooldown(0.1) {}

    void update(sf::Vector2f position, float dt, sf::Vector2f mousePos, bool mousePressed) override {
        togglePosition.update(dt);
        if (clickCooldown >= 0) clickCooldown -= dt;

        textObj->update(position + sf::Vector2f(leftPadding, topPadding), dt, mousePos, mousePressed);

        this->position = position;

        width = fmaxf(textObj->getWidth(), toggleWidth+2*padding) + leftPadding + rightPadding;
        height = textObj->getHeight() + toggleHeight + topPadding + bottomPadding + padding*2;

        toggleWidth = fmaxf(textObj->getWidth(), toggleWidth+2*padding) - 2*padding;

        handleInput(mousePos, mousePressed);
    }

    void handleInput(sf::Vector2f mousePos, bool mousePressed) {
        sf::Vector2f barPos = position + sf::Vector2f{leftPadding, topPadding} + sf::Vector2f{padding, textObj->getHeight() + padding};
        sf::Vector2f barEnd = barPos + sf::Vector2f{toggleWidth, toggleHeight};

        bool mouseOverBar = mousePos.x >= barPos.x && mousePos.x <= barEnd.x &&
                            mousePos.y >= barPos.y && mousePos.y <= barEnd.y;

        if (mousePressed && mouseOverBar && clickCooldown <= 0) {
            state = !state;

            togglePosition = state;

            clickCooldown = 0.1;
        }
    }


    void draw(sf::RenderTarget& target) const override {

        float rWidth = width - leftPadding - rightPadding;
        float rHeight = height - topPadding - bottomPadding;

        drawRoundedRectangle(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, bgColor, cornerRadius);
        drawRoundedOutline(target, position + sf::Vector2f{leftPadding, topPadding}, rWidth, rHeight, outlineColor, outlineColor, cornerRadius, outlineThickness);

        sf::Vector2f startPos = position + sf::Vector2f{leftPadding, topPadding} + sf::Vector2f{padding, textObj->getHeight() + padding};

        float t = togglePosition.get();
        sf::Color color1(57, 57, 57);
        sf::Color color2 = sf::Color::White;

        sf::Color c(
            static_cast<uint8_t>(color1.r * (1 - t) + color2.r * t),
            static_cast<uint8_t>(color1.g * (1 - t) + color2.g * t),
            static_cast<uint8_t>(color1.b * (1 - t) + color2.b * t),
            static_cast<uint8_t>(color1.a * (1 - t) + color2.a * t)
        );

        drawRoundedRectangle(target, startPos, toggleWidth, toggleHeight, c, cornerRadius-padding);
        drawRoundedRectangle(target, startPos+sf::Vector2f(outlineThickness + (toggleWidth/2-outlineThickness) * togglePosition.get(), outlineThickness), (toggleWidth-2*outlineThickness)/2, toggleHeight-2*outlineThickness, sf::Color{160, 160, 160}, cornerRadius-padding-outlineThickness);

        textObj->draw(target);
    }

    bool getValue() {return state;}
};

#endif //TOGGLE_H
