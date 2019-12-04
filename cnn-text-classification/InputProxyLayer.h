
#pragma once

#include "Layer.h"
#include "InputLayer.h"


class InputProxyLayer :  public Layer
{
public:
    InputProxyLayer(InputLayer *pInputLayer, std::string paramName);
    virtual ~InputProxyLayer();

public:
    virtual std::vector<int> GetOutputDimension();
    virtual std::vector<int> GetCurrentOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    InputLayer *m_pInputLayer;
    std::string m_paramName;
    std::vector<int> m_dim;

    float *m_Output;
};
