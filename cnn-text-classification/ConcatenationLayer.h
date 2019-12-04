
#pragma once

#include "Layer.h"


class ConcatenationLayer : public Layer
{
public:
    ConcatenationLayer(Layer **pInputs, int Count);
    virtual ~ConcatenationLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    std::vector<int> m_dims;
    int m_dimTotal;
    float *m_Output;
};

