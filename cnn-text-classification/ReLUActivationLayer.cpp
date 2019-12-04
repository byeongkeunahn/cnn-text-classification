
#include "pch.h"
#include "ReLUActivationLayer.h"


ReLUActivationLayer::ReLUActivationLayer(Layer *pInput)
{
    AddInLayer(pInput);
}

ReLUActivationLayer::~ReLUActivationLayer()
{
}

std::vector<int> ReLUActivationLayer::GetOutputDimension() {
    return m_inLayers[0]->GetOutputDimension();
}

std::vector<int> ReLUActivationLayer::GetCurrentOutputDimension() {
    return m_inLayers[0]->GetCurrentOutputDimension();
}

int ReLUActivationLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void ReLUActivationLayer::ForwardProp() {
    float *input = m_inLayers[0]->GetOutput();

    auto curDim = m_inLayers[0]->GetCurrentOutputDimension();
    int cnt = 1;
    for (auto d : curDim) cnt *= d;

    if (m_OutputBuffer.size() < cnt) {
        m_OutputBuffer.resize(cnt);
        m_OutputGradBuffer.resize(cnt);
        m_InputGradBuffer.resize(cnt);
    }

    for (int i = 0; i < cnt; i++) {
        m_OutputBuffer[i] = std::max(0.0f, input[i]);
    }

    m_OutputGrad = &m_OutputGradBuffer[0];
    memset(m_OutputGrad, 0, sizeof(float)*cnt);
}

void ReLUActivationLayer::BackwardProp() {
    auto curDim = m_inLayers[0]->GetCurrentOutputDimension();
    int cnt = 1;
    for (auto d : curDim) cnt *= d;

    /* compute the input gradient */
    for (int i = 0; i < cnt; i++) {
        m_InputGradBuffer[i] = m_OutputGrad[i] * ((m_OutputBuffer[i] > 0.0f) ? 1 : 0);
    }

    /* broadcasting */
    BroadcastInputGradient(&m_InputGradBuffer[0]);
}

float* ReLUActivationLayer::GetOutput() {
    return &m_OutputBuffer[0];
}

void ReLUActivationLayer::UpdateParams() {
    /* a ReLU activation layer does not have any associated parameters */
}
