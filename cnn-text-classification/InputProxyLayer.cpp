
#include "pch.h"
#include "InputProxyLayer.h"


InputProxyLayer::InputProxyLayer(InputLayer *pInputLayer, std::string paramName) {
    if (pInputLayer == nullptr || paramName.empty()) {
        throw std::exception("InputProxyLayer: Invalid parameters");
    }

    m_pInputLayer = pInputLayer;
    m_paramName = paramName;
    m_dim = pInputLayer->GetDataDimension(paramName.c_str());

    AddInLayer(pInputLayer);

    m_OutputGrad = nullptr; // this prevents unwanted backpropagation
}

InputProxyLayer::~InputProxyLayer() {
}

std::vector<int> InputProxyLayer::GetOutputDimension() {
    return std::vector<int>(m_dim);
}

std::vector<int> InputProxyLayer::GetCurrentOutputDimension() {
    return m_pInputLayer->GetCurrentDataDimension(m_paramName.c_str());
}

int InputProxyLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void InputProxyLayer::ForwardProp() {
    m_Output = m_pInputLayer->GetCurrentData(m_paramName.c_str());
}

void InputProxyLayer::BackwardProp() {
    /* do nothing, since the input should not be touched */
}

float* InputProxyLayer::GetOutput() {
    return m_Output;
}

void InputProxyLayer::UpdateParams() {
    /* an input-proxy layer does not have any associated parameters */
}
