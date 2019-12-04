
#pragma once

#include "Optimizer.h"


class AdadeltaOptimizer : public Optimizer
{
public:
    AdadeltaOptimizer(float decay_rate, float epsilon);
    virtual ~AdadeltaOptimizer();

public:
    virtual void Initialize(int numParams);
    virtual float Optimize(int idx, float prev_param, float new_grad);

private:
    float m_decay_rate, m_epsilon;
    std::vector<float> m_g2, m_dx2;
};
