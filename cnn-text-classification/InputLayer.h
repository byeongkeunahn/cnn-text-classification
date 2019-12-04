
#pragma once

#include "Layer.h"
#include "InputDataProvider.h"

class InputLayer : public Layer
{
public:
    InputLayer();
    virtual ~InputLayer();

public:
    float *GetCurrentData(const char *key);
    std::vector<int> GetDataDimension(const char *key);
    std::vector<int> GetCurrentDataDimension(const char *key);

public:
    virtual std::vector<int> GetOutputDimension();

    virtual int GetNumberOfParams();

    virtual void ForwardProp();
    virtual void BackwardProp();

    virtual float* GetOutput();
    virtual void UpdateParams();

public:
    void SetInputDataProvider(InputDataProvider *pProvider);

private:
    InputDataProvider *m_pProvider;
};

