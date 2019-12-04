
#pragma once


class Dataset
{
public:
    Dataset();
    virtual ~Dataset();

public:
    virtual size_t Count() = 0;
    virtual float *GetData(size_t idx, const char *key) = 0;
    virtual std::vector<int> GetCommonDataDimension(const char *key) = 0;
    virtual std::vector<int> GetDataDimension(size_t idx, const char *key) = 0;
};
