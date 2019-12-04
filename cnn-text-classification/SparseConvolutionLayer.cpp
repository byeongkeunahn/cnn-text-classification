
#include "pch.h"
#include "SparseConvolutionLayer.h"
#include "Random.h"


SparseConvolutionLayer::SparseConvolutionLayer(Layer *pInput, int H, int Mask) {
    auto indim = pInput->GetOutputDimension();
    if (indim[0] <= -2 || indim[0] == 0 || indim[1] <= 0) {
        throw std::exception("SparseConvolutionLayer: Invalid input layer dimension");
    }
    if (indim[0] >= 1 && (indim[0] - H + 1) <= 0) {
        throw std::exception("SparseConvolutionLayer: N - H + 1 <= 0");
    }

    m_N = indim[0];
    m_D = indim[1];
    m_H = H;
    m_EffectiveH = 0;
    m_Mask.resize(m_H);
    for (int i = 0; i < m_H; i++) {
        if (Mask & (1 << i)) {
            m_Mask[i] = true; // active
            m_EffectiveH++;
        }
        else {
            m_Mask[i] = false; // inactive
        }
    }
    if (m_EffectiveH == 0) {
        throw std::exception("SparseConvolutionLayer: Mask should specify at least one layer within the [0..H-1] range");
    }
    if (!m_Mask[0]) {
        throw std::exception("SparseConvolutionLayer: Mask should enable the zeroth vector");
    }

    m_W = new float*[m_H];
    m_W[0] = new float[m_D * m_EffectiveH];
    for (int i = 0, j = 0; i < m_H; i++) {
        if (m_Mask[i]) {
            m_W[i] = m_W[0] + (j++*m_D);
        }
        else {
            m_W[i] = nullptr;
        }
    }
    Random::Normal(sqrt(2.0f / (m_D*m_EffectiveH)), m_W[0], m_D*m_EffectiveH);

    m_WGrad = new float*[m_H];
    m_WGrad[0] = new float[m_D * m_EffectiveH];
    for (int i = 0, j = 0; i < m_H; i++) {
        if (m_Mask[i]) {
            m_WGrad[i] = m_WGrad[0] + (j++*m_D);
        }
        else {
            m_WGrad[i] = nullptr;
        }
    }
    memset(m_WGrad[0], 0, sizeof(float) * m_D * m_EffectiveH);

    m_b = 0;
    m_bGrad = 0;

    AddInLayer(pInput);

    m_steps = 0;
}

SparseConvolutionLayer::~SparseConvolutionLayer() {
}

std::vector<int> SparseConvolutionLayer::GetOutputDimension() {
    int outDim = -1;
    if (m_N != -1) outDim = m_N - m_H + 1;
    return std::vector<int>({ outDim });
}

std::vector<int> SparseConvolutionLayer::GetCurrentOutputDimension() {
    int curOutDim;
    if (m_N == -1) {
        int N = GetCurrentInputDimension()[0];
        curOutDim = std::max(1, N - m_H + 1);
    }
    else {
        curOutDim = m_N - m_H + 1;
    }
    return std::vector<int>({ curOutDim });
}

int SparseConvolutionLayer::GetNumberOfParams() {
    return m_D * m_EffectiveH + 1;
}

void SparseConvolutionLayer::ForwardProp() {
    /* resize buffers */
    int curOutDim = GetCurrentOutputDimension()[0];
    if (m_OutputBuffer.size() < curOutDim) {
        m_OutputBuffer.resize(curOutDim);
        m_OutputGradBuffer.resize(curOutDim);
    }
    auto curInDim = GetCurrentInputDimension();
    int curInCnt = curInDim[0] * curInDim[1];
    if (m_InputGradBuffer.size() < curInCnt) {
        m_InputGradBuffer.resize(curInCnt);
    }

    float *input = m_inLayers[0]->GetOutput();
    for (int t = 0; t < curOutDim; t++) {
        float value = m_b;
        int tmax = std::min(curInDim[0], t + m_H);
        for (int i = t; i < tmax; i++) {
            if (m_Mask[i - t]) {
                for (int j = 0; j < m_D; j++) {
                    value += input[i*m_D + j] * m_W[i - t][j];
                }
            }
        }
        m_OutputBuffer[t] = value;
    }

    m_OutputGrad = &m_OutputGradBuffer[0];
    memset(m_OutputGrad, 0, sizeof(float)*curOutDim);
}

void SparseConvolutionLayer::BackwardProp() {
    auto curInDim = GetCurrentInputDimension();
    int curOutDim = GetCurrentOutputDimension()[0];
    float *input = m_inLayers[0]->GetOutput();

    /* compute the parameter gradient */
    for (int t = 0; t < curOutDim; t++) {
        float outGrad = m_OutputGrad[t];
        if (std::abs(outGrad) < 1e-20) continue;

        /* for the weight matrix */
        int tmax = std::min(curInDim[0], t + m_H);
        for (int i = t; i < tmax; i++) {
            if (m_Mask[i - t]) {
                for (int j = 0; j < m_D; j++) {
                    m_WGrad[i - t][j] += input[i*m_D + j] * outGrad;
                }
            }
        }

        /* for the bias */
        m_bGrad += outGrad;
    }
    m_steps++;

    /* compute the input gradient */
    memset(&m_InputGradBuffer[0], 0, sizeof(float) * curInDim[0] * curInDim[1]);
    for (int t = 0; t < curOutDim; t++) {
        float outGrad = m_OutputGrad[t];
        if (std::abs(outGrad) < 1e-20) continue;

        int tmax = std::min(curInDim[0], t + m_H);
        for (int i = t; i < tmax; i++) {
            if (m_Mask[i - t]) {
                for (int j = 0; j < m_D; j++) {
                    m_InputGradBuffer[(i - t)*m_D + j] += m_W[i - t][j] * outGrad;
                }
            }
        }
    }

    /* broadcasting */
    BroadcastInputGradient(&m_InputGradBuffer[0]);
}

float* SparseConvolutionLayer::GetOutput() {
    return &m_OutputBuffer[0];
}

void SparseConvolutionLayer::UpdateParams() {
    /* optimize the weight matrix */
    float *W = m_W[0], *WGrad = m_WGrad[0];
    for (int i = 0; i < m_D * m_EffectiveH; i++) {
        W[i] = Optimize(i, W[i], WGrad[i] / m_steps);
    }
    memset(WGrad, 0, sizeof(float) * m_D * m_EffectiveH);

    /* optimize the bias */
    float &b = m_b, &bGrad = m_bGrad;
    b = Optimize(m_D*m_EffectiveH, b, bGrad);
    bGrad = 0;

    /* reset the number of accumulation steps */
    m_steps = 0;
}

std::vector<int> SparseConvolutionLayer::GetInputDimension() {
    return std::vector<int>({ m_N, m_D });
}

std::vector<int> SparseConvolutionLayer::GetCurrentInputDimension() {
    return m_inLayers[0]->GetCurrentOutputDimension();
}
