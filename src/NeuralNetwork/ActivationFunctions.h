//
// Created by Andreas Royset on 1/1/26.
//

#pragma once

#include <vector>
#include <cmath>
#include <algorithm>

class ActivationFunction {
public:
    virtual ~ActivationFunction() = default;

    [[nodiscard]] virtual float activate(float x) const = 0;

    [[nodiscard]] virtual float derivative(float x) const = 0;

    [[nodiscard]] virtual ActivationFunction* clone() const = 0;

    [[nodiscard]] virtual std::string getName() const = 0;
};

// ==================== ReLU ====================
class ReLU : public ActivationFunction {
public:
    [[nodiscard]] float activate(float x) const override {
        return std::max(0.0f, x);
    }

    [[nodiscard]] float derivative(float x) const override {
        return x > 0.0f ? 1.0f : 0.0f;
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new ReLU(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "ReLU";
    }
};

// ==================== Sigmoid ====================
class Sigmoid : public ActivationFunction {
public:
    [[nodiscard]] float activate(float x) const override {
        return 1.0f / (1.0f + std::exp(-x));
    }

    [[nodiscard]] float derivative(float x) const override {
        float sig = activate(x);
        return sig * (1.0f - sig);
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new Sigmoid(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "Sigmoid";
    }
};

// ==================== Tanh ====================
class Tanh : public ActivationFunction {
public:
    [[nodiscard]] float activate(float x) const override {
        return std::tanh(x);
    }

    [[nodiscard]] float derivative(float x) const override {
        float th = activate(x);
        return 1.0f - (th * th);
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new Tanh(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "Tanh";
    }
};

// ==================== Leaky ReLU ====================
class LeakyReLU : public ActivationFunction {
    float alpha;

public:
    explicit LeakyReLU(float alpha = 0.01f) : alpha(alpha) {}

    [[nodiscard]] float activate(float x) const override {
        return x > 0.0f ? x : alpha * x;
    }

    [[nodiscard]] float derivative(float x) const override {
        return x > 0.0f ? 1.0f : alpha;
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new LeakyReLU(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "LeakyReLU(" + std::to_string(alpha) + ")";
    }

    [[nodiscard]] float getAlpha() const { return alpha; }
};

// ==================== Linear ====================
class Linear : public ActivationFunction {
public:
    [[nodiscard]] float activate(float x) const override {
        return x;
    }

    [[nodiscard]] float derivative(float x) const override {
        return 1.0f;
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new Linear(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "Linear";
    }
};

// ==================== ELU ====================
class ELU : public ActivationFunction {
    float alpha;

public:
    explicit ELU(float alpha = 1.0f) : alpha(alpha) {}

    [[nodiscard]] float activate(float x) const override {
        return x >= 0.0f ? x : alpha * (std::exp(x) - 1.0f);
    }

    [[nodiscard]] float derivative(float x) const override {
        return x >= 0.0f ? 1.0f : alpha * std::exp(x);
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new ELU(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "ELU(" + std::to_string(alpha) + ")";
    }

    [[nodiscard]] float getAlpha() const { return alpha; }
};

// ==================== SELU ====================
class SELU : public ActivationFunction {
    static constexpr float LAMBDA = 1.0507f;
    static constexpr float ALPHA = 1.6733f;

public:
    [[nodiscard]] float activate(float x) const override {
        return LAMBDA * (x >= 0.0f ? x : ALPHA * (std::exp(x) - 1.0f));
    }

    [[nodiscard]] float derivative(float x) const override {
        return LAMBDA * (x >= 0.0f ? 1.0f : ALPHA * std::exp(x));
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new SELU(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "SELU";
    }
};

// ==================== STEP ====================
class STEP : public ActivationFunction {
    float alpha = 0;
    float beta = 1;

public:
    explicit STEP(float alpha = 0, float beta = 1) : alpha(alpha), beta(beta) {}

    [[nodiscard]] float activate(float x) const override {
        return x < 0 ? alpha : beta;
    }

    [[nodiscard]] float derivative(float x) const override {
        return 0;
    }

    [[nodiscard]] ActivationFunction* clone() const override {
        return new STEP(*this);
    }

    [[nodiscard]] std::string getName() const override {
        return "STEP";
    }
};