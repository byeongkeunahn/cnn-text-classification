
#pragma once

#include "Layer.h"


class MaxPoolingLayer : public Layer
{
public:
    MaxPoolingLayer(Layer *pInput);
    virtual ~MaxPoolingLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    int m_dim, m_dimCurrent;
    int m_ArgMaxIndex;
    float m_Output;
    std::vector<float> m_InputGrad;
};

