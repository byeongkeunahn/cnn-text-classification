
#pragma once

#include "Layer.h"
#include "EmbeddingLayer.h"


class Sentence2DLayer : public Layer
{
public:
    Sentence2DLayer(EmbeddingLayer *pEmbeddings, Layer *pWordSequence, bool fTrainEmbeddings);
    virtual ~Sentence2DLayer();

public:
    virtual std::vector<int> GetOutputDimension();
    virtual std::vector<int> GetCurrentOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

private:
    EmbeddingLayer *m_pEmbeddings;
    Layer *m_pWordSequence;
    bool m_fTrainEmbeddings;

    int m_indim, m_WordCount, m_D;

    std::vector<float> m_OutputBuffer, m_OutputGradBuffer;
};

