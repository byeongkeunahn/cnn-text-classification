
#pragma once

#include "Layer.h"


class DropoutLayer : public Layer
{
public:
    DropoutLayer(Layer *pInput, float keep_prob);
    virtual ~DropoutLayer();

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
    float *m_fKeepMap;
    float m_keepProb;
};

