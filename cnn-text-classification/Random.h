
#pragma once


class Random
{
public:
    static float Uniform();
    static void Xavier(int indim, int outdim, float **W); // samples from N(0, 1/dim)
    static void Normal(float stdev, float *buf, int count);
};

