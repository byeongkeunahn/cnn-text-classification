
#pragma once

#include "Dataset.h"


class InputDataProvider
{
public:
    InputDataProvider(Dataset *pDataset, size_t idx);
    ~InputDataProvider();

public:
    float *GetCurrentData(const char *key);
    std::vector<int> GetDataDimension(const char *key);
    std::vector<int> GetCurrentDataDimension(const char *key);

private:
    Dataset *m_pDataset;
    size_t m_idx;
};
