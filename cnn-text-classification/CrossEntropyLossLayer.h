
#pragma once

#include "Layer.h"


class CrossEntropyLossLayer : public Layer
{
public:
    CrossEntropyLossLayer(Layer *pProbDist, Layer *pTrueLabelOneHot);
    virtual ~CrossEntropyLossLayer();

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    int m_dim;
    float m_Output, *m_InputGrad;
    Layer *m_pProbDist, *m_pTrueLabelOneHot;
};

