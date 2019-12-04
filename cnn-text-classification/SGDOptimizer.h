
#pragma once

#include "Optimizer.h"


class SGDOptimizer : public Optimizer
{
public:
    SGDOptimizer();
    virtual ~SGDOptimizer();

public:
    virtual void Initialize(int numParams);
    virtual float Optimize(int idx, float prev_param, float new_grad);

    void SetLearningRate(float learning_rate);

private:
    float m_learning_rate;
};
