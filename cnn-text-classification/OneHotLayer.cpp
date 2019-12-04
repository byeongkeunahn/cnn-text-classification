
#include "pch.h"
#include "OneHotLayer.h"


OneHotLayer::OneHotLayer(Layer *pInput, std::vector<int> labels) {
    auto indim = pInput->GetOutputDimension();
    if (indim.size() != 1 || indim[0] != 1) {
        throw std::exception("OneHotLayer: Invalid input layer size (!= 1)");
    }

    if (labels.size() <= 1) {
        throw std::exception("OneHotLayer: Insufficient number of labels");
    }

    for (int i : labels) m_labels.push_back(i);
    std::sort(m_labels.begin(), m_labels.end());
    for (size_t i = 1; i < m_labels.size(); i++) {
        if (m_labels[i - 1] == m_labels[i]) {
            throw std::exception("OneHotLayer: Duplicate labels");
        }
    }
    m_labels.clear();
    for (int i : labels) m_labels.push_back(i);

    m_dim = (int)m_labels.size();
    m_OutputBuffer.resize(m_dim);

    AddInLayer(pInput);

    m_OutputGrad = nullptr;
}

OneHotLayer::~OneHotLayer() {
}

std::vector<int> OneHotLayer::GetOutputDimension() {
    return std::vector<int>({ m_dim });
}

int OneHotLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void OneHotLayer::ForwardProp() {
    memset(&m_OutputBuffer[0], 0, sizeof(float) * m_dim);

    float inputValue = m_inLayers[0]->GetOutput()[0];
    for (int i = 0; i < m_dim; i++) {
        if (std::abs(inputValue - m_labels[i]) < 1e-20) {
            m_OutputBuffer[i] = 1;
            break;
        }
    }
}

void OneHotLayer::BackwardProp() {
    /* no backward propagation needed */
}

float* OneHotLayer::GetOutput() {
    return &m_OutputBuffer[0];
}

void OneHotLayer::UpdateParams() {
    /* a one-hot layer does not have any associated parameters */
}
