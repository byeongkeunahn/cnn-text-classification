
#include "pch.h"
#include "InputDataProvider.h"


InputDataProvider::InputDataProvider(Dataset *pDataset, size_t idx) {
    if (idx >= pDataset->Count()) {
        throw std::exception("InputDataProvider: idx is out of range");
    }
    m_pDataset = pDataset;
    m_idx = idx;
}

InputDataProvider::~InputDataProvider() {
}

float *InputDataProvider::GetCurrentData(const char *key) {
    return m_pDataset->GetData(m_idx, key);
}

std::vector<int> InputDataProvider::GetDataDimension(const char *key) {
    return m_pDataset->GetCommonDataDimension(key);
}

std::vector<int> InputDataProvider::GetCurrentDataDimension(const char *key) {
    return m_pDataset->GetDataDimension(m_idx, key);
}
