
#include "pch.h"
#include "Random.h"


static std::default_random_engine s_generator(1); // **NOT** THREAD-SAFE

float Random::Uniform()
{
    std::uniform_real_distribution<float> dist(0, 1);
    return dist(s_generator);
}

void Random::Xavier(int indim, int outdim, float **W) {
    std::normal_distribution<float> dist(0, 2.0f/sqrtf(indim + outdim));
    for (int i = 0; i < outdim; i++) {
        for (int j = 0; j < indim; j++) {
            W[i][j] = dist(s_generator);
        }
    }
}

void Random::Normal(float stdev, float *buf, int count) {
    std::normal_distribution<float> dist(0, stdev);
    for (int i = 0; i < count; i++) {
        buf[i] = dist(s_generator);
    }
}
