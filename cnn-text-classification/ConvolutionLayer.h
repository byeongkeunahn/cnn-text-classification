
#pragma once

#include "Layer.h"


class ConvolutionLayer : public Layer
{
public:
    ConvolutionLayer(Layer *pInput, int H);
    ConvolutionLayer(const ConvolutionLayer&) = delete;
    virtual ~ConvolutionLayer();

public:
    virtual std::vector<int> GetOutputDimension();
    virtual std::vector<int> GetCurrentOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    std::vector<int> GetInputDimension();
    std::vector<int> GetCurrentInputDimension();

private:
    int m_N, m_D, m_H;
    std::vector<float> m_OutputBuffer, m_OutputGradBuffer, m_InputGradBuffer;
    float **m_W, **m_WGrad;
    float m_b, m_bGrad;
    int m_steps;
};

