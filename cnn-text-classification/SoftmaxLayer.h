
#pragma once

#include "Layer.h"


class SoftmaxLayer : public Layer
{
public:
    SoftmaxLayer(Layer *pInput);
    virtual ~SoftmaxLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    int m_dim;
    float *m_Output, *m_InputGrad;
};

