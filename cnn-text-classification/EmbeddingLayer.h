
#pragma once

#include "Layer.h"


class EmbeddingLayer : public Layer
{
public:
    EmbeddingLayer(const wchar_t *lpszFilePath);
    virtual ~EmbeddingLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

    void UpdateOutputGradientSingleWord(int wid, float *embed_grad);

private:
    int m_N, m_D;
    float **m_W;
    int m_steps;
};

