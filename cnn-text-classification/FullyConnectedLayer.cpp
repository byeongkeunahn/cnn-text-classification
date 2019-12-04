
#include "pch.h"
#include "FullyConnectedLayer.h"
#include "Random.h"


FullyConnectedLayer::FullyConnectedLayer(Layer *pInput, int OutDim)
{
    auto indim = pInput->GetOutputDimension();
    if (indim.size() != 1 || indim[0] <= 0) {
        throw std::exception("FullyConnectedLayer: Input layer has wrong dimensions");
    }
    m_indim = indim[0];
    if (OutDim <= 0) {
        throw std::exception("FullyConnectedLayer: OutDim <= 0 is an invalid hyperparameter");
    }
    m_outdim = OutDim;

    m_Output = new float[m_outdim];
    m_OutputGrad = new float[m_outdim];
    m_InputGrad = new float[m_indim];

    m_W = new float*[m_outdim];
    m_W[0] = new float[m_indim * m_outdim];
    for (int i = 0; i < m_outdim; i++) {
        m_W[i] = m_W[0] + (i*m_indim);
    }
    Random::Xavier(m_indim, m_outdim, m_W);

    m_WGrad = new float*[m_outdim];
    m_WGrad[0] = new float[m_indim * m_outdim];
    for (int i = 0; i < m_outdim; i++) {
        m_WGrad[i] = m_WGrad[0] + (i*m_indim);
    }
    memset(m_WGrad[0], 0, sizeof(float) * m_indim * m_outdim);

    m_b = new float[m_outdim];
    m_bGrad = new float[m_outdim];
    memset(m_b, 0, sizeof(float) * m_outdim);
    memset(m_bGrad, 0, sizeof(float) * m_outdim);

    m_steps = 0;

    AddInLayer(pInput);
}

FullyConnectedLayer::~FullyConnectedLayer()
{
    delete[] m_Output;
    delete[] m_OutputGrad;
    delete[] m_InputGrad;
    delete[] m_W[0];
    delete[] m_W;
    delete[] m_WGrad[0];
    delete[] m_WGrad;
    delete[] m_b;
    delete[] m_bGrad;
}

std::vector<int> FullyConnectedLayer::GetOutputDimension() {
    return std::vector<int>({ m_outdim });
}

int FullyConnectedLayer::GetNumberOfParams() {
    return m_indim * m_outdim + m_outdim;
}

void FullyConnectedLayer::ForwardProp() {
    Layer *pInputLayer = m_inLayers[0]; // there is only one such layer
    float *x = (float *)pInputLayer->GetOutput();

    /* matrix multiplication */
    float *y = m_Output, **W = m_W, *b = m_b;
    for (int i = 0; i < m_outdim; i++) {
        y[i] = b[i];
        for (int j = 0; j < m_indim; j++) {
            y[i] += W[i][j] * x[j];
        }
    }

    memset(m_OutputGrad, 0, sizeof(float)*m_outdim);
}

void FullyConnectedLayer::BackwardProp() {
    Layer *pInputLayer = m_inLayers[0]; // there is only one such layer

    /* compute the parameter gradient: for weight matrix */
    float *x = (float *)pInputLayer->GetOutput();
    for (int i = 0; i < m_outdim; i++) {
        for (int j = 0; j < m_indim; j++) {
            m_WGrad[i][j] += x[j] * m_OutputGrad[i];
        }
    }
    /* compute the parameter gradient: for bias */
    for (int i = 0; i < m_outdim; i++) {
        m_bGrad[i] += m_OutputGrad[i];
    }
    m_steps++;

    /* compute the input gradient */
    float **W = m_W;
    memset(m_InputGrad, 0, sizeof(float) * m_indim);
    for (int i = 0; i < m_outdim; i++) {
        for (int j = 0; j < m_indim; j++) {
            m_InputGrad[j] += W[i][j] * m_OutputGrad[i];
        }
    }
}

float* FullyConnectedLayer::GetOutput() {
    return m_Output;
}

void FullyConnectedLayer::UpdateParams() {
    /* optimize the weight matrix */
    float *W = m_W[0], *WGrad = m_WGrad[0];
    for (int i = 0; i < m_indim * m_outdim; i++) {
        W[i] = Optimize(i, W[i], WGrad[i] / m_steps);
    }
    memset(WGrad, 0, sizeof(float) * m_indim * m_outdim);

    /* optimize the bias */
    float *b = m_b, *bGrad = m_bGrad;
    for (int i = 0; i < m_outdim; i++) {
        b[i] = Optimize(m_indim * m_outdim + i, b[i], bGrad[i] / m_steps);
    }
    memset(bGrad, 0, sizeof(float) * m_outdim);

    /* reset the number of accumulation steps */
    m_steps = 0;
}
