
#pragma once

#include "Layer.h"


class FullyConnectedLayer : public Layer
{
public:
    FullyConnectedLayer(Layer *pInput, int OutDim);
    virtual ~FullyConnectedLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    int m_indim, m_outdim;
    float *m_Output, *m_InputGrad;
    float **m_W, **m_WGrad;
    float *m_b, *m_bGrad;
    int m_steps;
};

