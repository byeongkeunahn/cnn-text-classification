
#pragma once

#include "Layer.h"


class ReLUActivationLayer : public Layer
{
public:
    ReLUActivationLayer(Layer *pInput);
    virtual ~ReLUActivationLayer();

public:
    virtual std::vector<int> GetOutputDimension();
    virtual std::vector<int> GetCurrentOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    std::vector<float> m_OutputBuffer;
    std::vector<float> m_OutputGradBuffer;
    std::vector<float> m_InputGradBuffer;
};

