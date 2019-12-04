
#pragma once


class Optimizer
{
public:
    Optimizer();
    virtual ~Optimizer();

public:
    virtual void Initialize(int numParams) = 0;
    virtual float Optimize(int idx, float prev_param, float new_grad) = 0;
};

