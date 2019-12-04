
#include "pch.h"
#include "ConvolutionLayer.h"
#include "Random.h"


ConvolutionLayer::ConvolutionLayer(Layer *pInput, int H) {
    auto indim = pInput->GetOutputDimension();
    if (indim[0] <= -2 || indim[0] == 0 || indim[1] <= 0) {
        throw std::exception("ConvolutionLayer: Invalid input layer dimension");
    }
    if (indim[0] >= 1 && (indim[0] - H + 1) <= 0) {
        throw std::exception("ConvolutionLayer: N - H + 1 <= 0");
    }

    m_N = indim[0];
    m_D = indim[1];
    m_H = H;

    m_W = new float*[m_H];
    m_W[0] = new float[m_D * m_H];
    for (int i = 0; i < m_H; i++) {
        m_W[i] = m_W[0] + (i*m_D);
    }
    Random::Normal(sqrt(2.0f / (m_D*m_H)), m_W[0], m_D*m_H);

    m_WGrad = new float*[m_H];
    m_WGrad[0] = new float[m_D * m_H];
    for (int i = 0; i < m_H; i++) {
        m_WGrad[i] = m_WGrad[0] + (i*m_D);
    }
    memset(m_WGrad[0], 0, sizeof(float) * m_D * m_H);

    m_b = 0;
    m_bGrad = 0;

    AddInLayer(pInput);

    m_steps = 0;
}

ConvolutionLayer::~ConvolutionLayer() {
}

std::vector<int> ConvolutionLayer::GetOutputDimension() {
    int outDim = -1;
    if (m_N != -1) outDim = m_N - m_H + 1;
    return std::vector<int>({ outDim });
}

std::vector<int> ConvolutionLayer::GetCurrentOutputDimension() {
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

int ConvolutionLayer::GetNumberOfParams() {
    return m_D * m_H + 1;
}

void ConvolutionLayer::ForwardProp() {
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
            for (int j = 0; j < m_D; j++) {
                value += input[i*m_D + j] * m_W[i - t][j];
            }
        }
        m_OutputBuffer[t] = value;
    }

    m_OutputGrad = &m_OutputGradBuffer[0];
    memset(m_OutputGrad, 0, sizeof(float)*curOutDim);
}

void ConvolutionLayer::BackwardProp() {
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
            for (int j = 0; j < m_D; j++) {
                m_WGrad[i - t][j] += input[i*m_D + j] * outGrad;
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
            for (int j = 0; j < m_D; j++) {
                m_InputGradBuffer[(i - t)*m_D + j] += m_W[i-t][j] * outGrad;
            }
        }
    }

    /* broadcasting */
    BroadcastInputGradient(&m_InputGradBuffer[0]);
}

float* ConvolutionLayer::GetOutput() {
    return &m_OutputBuffer[0];
}

void ConvolutionLayer::UpdateParams() {
    /* optimize the weight matrix */
    float *W = m_W[0], *WGrad = m_WGrad[0];
    for (int i = 0; i < m_D * m_H; i++) {
        W[i] = Optimize(i, W[i], WGrad[i] / m_steps);
    }
    memset(WGrad, 0, sizeof(float) * m_D * m_H);

    /* optimize the bias */
    float &b = m_b, &bGrad = m_bGrad;
    b = Optimize(m_D*m_H, b, bGrad);
    bGrad = 0;

    /* reset the number of accumulation steps */
    m_steps = 0;
}

std::vector<int> ConvolutionLayer::GetInputDimension() {
    return std::vector<int>({ m_N, m_D });
}

std::vector<int> ConvolutionLayer::GetCurrentInputDimension() {
    return m_inLayers[0]->GetCurrentOutputDimension();
}
