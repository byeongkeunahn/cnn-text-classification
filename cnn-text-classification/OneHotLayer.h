
#pragma once

#include "Layer.h"


class OneHotLayer : public Layer
{
public:
    OneHotLayer(Layer *pInput, std::vector<int> labels);
    virtual ~OneHotLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    std::vector<int> m_labels;
    std::vector<float> m_OutputBuffer;
    int m_dim;
};

