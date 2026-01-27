//
// Created by Andreas Royset on 1/23/26.
//
#pragma once

#include <SFML/Graphics.hpp>

inline void drawRoundedRectangle(sf::RenderTarget& target, sf::Vector2f position, float width, float height, sf::Color color, float cornerRadius, int quality = 64) {
	if (width < 0 || height < 0) return;

	// Clamp corner radius to at most half the smaller dimension
	float maxRadius = fminf(width, height) / 2.0f;
	cornerRadius = fminf(cornerRadius, maxRadius);
	cornerRadius = fmaxf(cornerRadius, 0.00001f);

	sf::VertexArray vertices(sf::TriangleFan, quality+1);

	for (int i = 0; i < quality; i++) {
		float angle = float(i) / float(quality) * 2.0f * float(M_PI);
		sf::Vector2f pos = {cos(angle)*cornerRadius, sin(angle)*cornerRadius};

		if (pos.x < 0) pos.x += cornerRadius;
		else pos.x += width - cornerRadius;
		if (pos.y < 0) pos.y += cornerRadius;
		else pos.y += height - cornerRadius;

		pos += position;

		vertices[i].position = pos;
		vertices[i].color = color;
	}

	vertices[quality].position = vertices[0].position;
	vertices[quality].color = vertices[0].color;

	target.draw(vertices);
}

inline void drawRoundedOutline(sf::RenderTarget& target, sf::Vector2f position, float width, float height, sf::Color color1, sf::Color color2, float cornerRadius, float thickness, int quality = 64) {
	if (width < 0 || height < 0) return;

	// Clamp corner radius to at most half the smaller dimension
	float maxRadius = fminf(width, height) / 2.0f;
	cornerRadius = fminf(cornerRadius, maxRadius);
	cornerRadius = fmaxf(cornerRadius, 0.00001f);

	sf::VertexArray vertices(sf::TriangleStrip, 2*quality+2);

	for (int i = 0; i < quality; i++) {
		float angle = float(i) / float(quality) * 2.0f * float(M_PI);
		sf::Vector2f pos1 = {cos(angle)*cornerRadius, sin(angle)*cornerRadius};

		if (pos1.x < 0) pos1.x += cornerRadius;
		else pos1.x += width - cornerRadius;
		if (pos1.y < 0) pos1.y += cornerRadius;
		else pos1.y += height - cornerRadius;

		pos1 += position;

		vertices[i*2].position = pos1;
		vertices[i*2].color = color1;

		sf::Vector2f pos2 = {cos(angle)*(cornerRadius-thickness), sin(angle)*(cornerRadius-thickness)};

		if (pos2.x < 0) pos2.x += cornerRadius;
		else pos2.x += width - cornerRadius;
		if (pos2.y < 0) pos2.y += cornerRadius;
		else pos2.y += height - cornerRadius;

		pos2 += position;

		vertices[i*2+1].position = pos2;
		vertices[i*2+1].color = color2;
	}

	vertices[2*quality].position = vertices[0].position;
	vertices[2*quality].color = vertices[0].color;

	vertices[2*quality+1].position = vertices[1].position;
	vertices[2*quality+1].color = vertices[1].color;

	target.draw(vertices);
}

inline void drawThickLine(sf::RenderTarget& target, sf::Vector2f a, sf::Vector2f b, float thickness, sf::Color color) {
	sf::Vector2f d = b - a;
	float len = std::sqrt(d.x * d.x + d.y * d.y);
	if (len == 0.f) return;

	sf::Vector2f n(-d.y / len, d.x / len);   // perpendicular normal
	sf::Vector2f off = n * (thickness * 0.5f);

	sf::VertexArray quad(sf::Triangles, 6);

	quad[0] = sf::Vertex(a + off, color);
	quad[1] = sf::Vertex(b + off, color);
	quad[2] = sf::Vertex(b - off, color);

	quad[3] = sf::Vertex(a + off, color);
	quad[4] = sf::Vertex(b - off, color);
	quad[5] = sf::Vertex(a - off, color);

	target.draw(quad);
}