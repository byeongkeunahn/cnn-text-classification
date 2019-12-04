
#include "pch.h"
#include "EmbeddingLayer.h"


EmbeddingLayer::EmbeddingLayer(const wchar_t *lpszFilePath) {
    std::ifstream f(lpszFilePath);

    f >> m_N >> m_D;
    if (m_N <= 0 || m_D <= 0) {
        throw std::exception("EmbeddingLayer: Input file has invalid parameters");
    }

    m_W = new float*[m_N];
    m_W[0] = new float[m_N * m_D];
    for (int i = 0; i < m_N; i++) {
        m_W[i] = m_W[0] + (i*m_D);
    }

    /* read embeddings from file */
    for (int i = 0; i < m_N; i++) {
        for (int j = 0; j < m_D; j++) {
            f >> m_W[i][j];
        }
    }

    m_OutputGrad = new float[m_N * m_D];
    memset(m_OutputGrad, 0, sizeof(float)*m_N*m_D);
    m_steps = 0;
}

EmbeddingLayer::~EmbeddingLayer() {
    delete[] m_W[0];
    delete[] m_W;
}

std::vector<int> EmbeddingLayer::GetOutputDimension() {
    return std::vector<int>({ m_N, m_D });
}

int EmbeddingLayer::GetNumberOfParams() {
    return m_N * m_D;
}

void EmbeddingLayer::ForwardProp() {
    /* nothing to do */
}

void EmbeddingLayer::BackwardProp() {
    m_steps++;
}

float* EmbeddingLayer::GetOutput() {
    return m_W[0];
}

void EmbeddingLayer::UpdateParams() {
    float *W = m_W[0], *WGrad = m_OutputGrad;
    for (int i = 0; i < m_N * m_D; i++) {
        W[i] = Optimize(i, W[i], WGrad[i] / m_steps);
    }
    memset(WGrad, 0, sizeof(float) * m_N * m_D);
    m_steps = 0;
}

void EmbeddingLayer::UpdateOutputGradientSingleWord(int wid, float *embed_grad) {
    float *WGrad_wid = m_OutputGrad + (wid*m_D);
    for (int j = 0; j < m_D; j++) {
        WGrad_wid[j] += embed_grad[j];
    }
}
