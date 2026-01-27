//
// Created by Andreas Royset on 1/23/26.
//

#ifndef INTERPOLATED_H
#define INTERPOLATED_H
#include <cmath>

enum InterpolationFunction {
	LINEAR,
	EASE_IN_OUT_CUBIC,
	EASE_IN_BACK,
	EASE_OUT_BACK,
	EASE_OUT_ELASTIC
};

inline float linear(float x) {return x;}
inline float easeInOutCubic(float x) {return -2*x*x*x+3*x*x;}
inline float easeInBack(float x) {return 2.5f*x*x*x-1.5f*x*x;}
inline float easeOutBack(float x) {return 2.5f*std::pow(x-1, 3) + 1.5f*std::pow(x-1, 2) + 1;}
inline float easeOutElastic(float x) {
	const float c4 = (2 * M_PI) / 3;

	return x == 0 ? 0 : x == 1 ? 1: std::pow(2, -10 * x) * std::sin((x * 10 - 0.75) * c4) + 1;}

template <typename T>

class Interpolated {
	T currentValue;
	T targetValue;
	T startValue;

	float t;
	float time;

	InterpolationFunction function;

	public:

	Interpolated() : currentValue(), targetValue(), startValue(), t(), function(EASE_IN_OUT_CUBIC), time(1) {}
	Interpolated(T value, float time = 1, InterpolationFunction f = EASE_IN_OUT_CUBIC) : currentValue(value), targetValue(value), startValue(value), t(0), time(time), function(f) {}

	void operator = (T value) {
		set(value);
	}

	T operator + (T value) {
		return currentValue + value;
	}
	T operator - (T value) {
		return currentValue - value;
	}
	T operator * (T value) {
		return currentValue * value;
	}
	T operator / (T value) {
		return currentValue / value;
	}

	void operator += (T value) {
		set(currentValue + value);
	}
	void operator -= (T value) {
		set(currentValue - value);
	}
	void operator *= (T value) {
		set(currentValue * value);
	}
	void operator /= (T value) {
		set(currentValue / value);
	}

	bool operator == (T value) {
		return currentValue == value;
	}
	bool operator != (T value) {
		return currentValue != value;
	}
	bool operator > (T value) {
		return currentValue > value;
	}
	bool operator < (T value) {
		return currentValue < value;
	}
	bool operator >= (T value) {
		return currentValue >= value;
	}
	bool operator <= (T value) {
		return currentValue <= value;
	}

	explicit operator T() {return currentValue;}

	void reset() {targetValue = currentValue; startValue = currentValue; t = 0;}

	void set(T newValue) {targetValue = newValue; startValue = currentValue; t = 0; }

	T get() const {return currentValue;}

	bool update(float dt) {
		if (t <= 1 && currentValue != targetValue) {
			t += dt/time;

			float tValue = 0;

			switch(function) {
				case LINEAR:
					tValue = linear(t);
					break;
				case EASE_IN_OUT_CUBIC:
					tValue = easeInOutCubic(t);
					break;
				case EASE_IN_BACK:
					tValue = easeInBack(t);
					break;
				case EASE_OUT_BACK:
					tValue = easeOutBack(t);
					break;
				case EASE_OUT_ELASTIC:
					tValue = easeOutElastic(t);
					break;
			}

			currentValue = startValue * (1-tValue) + targetValue * tValue;
		} else if (t >= 0.5) {
			t = 0;
			currentValue = targetValue;
			startValue = currentValue;

			return true;
		}
		return false;
	}

	void setTime(float newTime) {time = newTime;}
};

#endif //INTERPOLATED_H
