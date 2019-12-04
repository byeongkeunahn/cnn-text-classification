
#pragma once

#include "Optimizer.h"


class Layer
{
public:
    Layer();
    virtual ~Layer();

public:
    virtual std::vector<int> GetOutputDimension() = 0;
    virtual std::vector<int> GetCurrentOutputDimension(); // override if you need (default behavior: static)

    virtual int GetNumberOfParams() = 0;
    virtual void SetOptimizer(Optimizer *pOptimizer, int nStartIndex);
    virtual void SetTesting(bool fTesting);

    virtual void ForwardProp() = 0;
    virtual void BackwardProp() = 0;

    virtual float* GetOutput() = 0; // flattened gradient. call after ForwardProp()
    virtual void UpdateOutputGradient(float *buf);
    virtual void UpdateParams() = 0;

    virtual Layer **GetInLayers();
    virtual size_t GetInLayersCount();

protected:
    float Optimize(int idx, float prev_param, float new_grad); // returns the new value for the variable **idx** (idx starts from 0)
    virtual void AddInLayer(Layer *pLayer); // keeps the order the layers are added
    virtual void BroadcastInputGradient(float *input_grad);

    std::vector<Layer*> m_inLayers;
    bool m_isTesting;

    float *m_OutputGrad;

private:
    Optimizer *m_pOptimizer;
    int m_nOptStartIndex;
};

