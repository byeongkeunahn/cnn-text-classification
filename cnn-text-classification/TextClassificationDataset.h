
#pragma once

#include "Dataset.h"


class TextClassificationDataset : public Dataset
{
public:
    TextClassificationDataset(const wchar_t *lpszSentencePath, const wchar_t *lpszLabelPath);
    virtual ~TextClassificationDataset();

public:
    virtual size_t Count();
    virtual float *GetData(size_t idx, const char *key);
    virtual std::vector<int> GetCommonDataDimension(const char *key);
    virtual std::vector<int> GetDataDimension(size_t idx, const char *key);

private:
    void ReadSentences(const wchar_t *lpszSentencePath);
    void ReadLabels(const wchar_t *lpszLabelPath);

private:
    std::vector<std::vector<float>> m_seqs;
    std::vector<float> m_labels;
};
