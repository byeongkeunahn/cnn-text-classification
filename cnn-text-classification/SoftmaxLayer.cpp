
#include "pch.h"
#include "SoftmaxLayer.h"
#include "Random.h"


SoftmaxLayer::SoftmaxLayer(Layer *pInput)
{
    auto indim = pInput->GetOutputDimension();
    if (indim.size() != 1 || indim[0] <= 0) {
        throw std::exception("SoftmaxLayer: Input layer has wrong dimensions");
    }
    m_dim = indim[0];
    m_Output = new float[m_dim];
    m_OutputGrad = new float[m_dim];
    m_InputGrad = new float[m_dim];

    AddInLayer(pInput);
}

SoftmaxLayer::~SoftmaxLayer()
{
    delete[] m_Output;
    delete[] m_OutputGrad;
    delete[] m_InputGrad;
}

std::vector<int> SoftmaxLayer::GetOutputDimension() {
    return std::vector<int>({ m_dim });
}

int SoftmaxLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void SoftmaxLayer::ForwardProp() {
    Layer *pInputLayer = m_inLayers[0]; // there is only one such layer
    float *input = (float *)pInputLayer->GetOutput();

    float max_val = input[0];
    for (int i = 0; i < m_dim; i++) {
        max_val = std::max(max_val, input[i]);
    }

    float Z = 0;
    for (int i = 0; i < m_dim; i++) {
        m_Output[i] = std::exp(input[i] - max_val);
        Z += m_Output[i];
    }
    for (int i = 0; i < m_dim; i++) {
        m_Output[i] /= Z;
    }

    memset(m_OutputGrad, 0, sizeof(float)*m_dim);
}

void SoftmaxLayer::BackwardProp() {
    /* compute the input gradient */
    float *dx = m_InputGrad, *dy = m_OutputGrad, *y = m_Output;
    for (int i = 0; i < m_dim; i++) {
        dx[i] = 0;
        for (int j = 0; j < m_dim; j++) {
            float Dij = (i == j) ? 1.0f : 0.0f;
            dx[i] += dy[j] * (Dij - y[i]) * y[j];
        }
    }

    /* broadcasting */
    BroadcastInputGradient(m_InputGrad);
}

float* SoftmaxLayer::GetOutput() {
    return m_Output;
}

void SoftmaxLayer::UpdateParams() {
    /* a softmax layer does not have any associated parameters */
}
