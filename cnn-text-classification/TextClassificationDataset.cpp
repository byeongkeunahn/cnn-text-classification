
#include "pch.h"
#include "TextClassificationDataset.h"


TextClassificationDataset::TextClassificationDataset(const wchar_t *lpszSentencePath, const wchar_t *lpszLabelPath) {
    ReadSentences(lpszSentencePath);
    ReadLabels(lpszLabelPath);

    if (m_seqs.size() != m_labels.size()) {
        throw std::exception("TextClassificationDataset: Sentence Count != Label Count");
    }
}

TextClassificationDataset::~TextClassificationDataset() {
}

size_t TextClassificationDataset::Count() {
    return m_seqs.size();
}

float *TextClassificationDataset::GetData(size_t idx, const char *key) {
    if (!strcmp(key, "seqs")) {
        return &m_seqs[idx][0];
    }
    if (!strcmp(key, "labels")) {
        return &m_labels[idx];
    }
    return nullptr;
}

std::vector<int> TextClassificationDataset::GetCommonDataDimension(const char *key) {
    if (!strcmp(key, "seqs")) {
        return std::vector<int>({ -1 });
    }
    if (!strcmp(key, "labels")) {
        return std::vector<int>({ 1 });
    }
    return std::vector<int>();
}

std::vector<int> TextClassificationDataset::GetDataDimension(size_t idx, const char *key) {
    if (!strcmp(key, "seqs")) {
        return std::vector<int>({ (int)m_seqs[idx].size() });
    }
    if (!strcmp(key, "labels")) {
        return std::vector<int>({ 1 });
    }
    return std::vector<int>();
}

void TextClassificationDataset::ReadSentences(const wchar_t *lpszSentencePath) {
    std::ifstream f(lpszSentencePath);
    int wid;
    while (1) {
        f >> wid;
        if (wid == -2) break;

        m_seqs.push_back(std::vector<float>());
        auto &vec = *m_seqs.rbegin();
        while (wid != -1)
        do {
            vec.push_back(wid);
            f >> wid;
        } while (wid != -1);
    }
}

void TextClassificationDataset::ReadLabels(const wchar_t *lpszLabelPath) {
    std::ifstream f(lpszLabelPath);
    int label;
    while (1) {
        f >> label;
        if (label == -2) break;

        m_labels.push_back((label == 1) ? 0 : 1);
    }
}
