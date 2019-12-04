
#include "pch.h"
#include "Layer.h"


Layer::Layer()
{
    m_isTesting = false;
}

Layer::~Layer()
{
}

std::vector<int> Layer::GetCurrentOutputDimension() {
    return GetOutputDimension();
}

void Layer::SetOptimizer(Optimizer *pOptimizer, int nStartIndex) {
    m_pOptimizer = pOptimizer;
    m_nOptStartIndex = nStartIndex;
}

void Layer::SetTesting(bool fTesting) {
    m_isTesting = fTesting;
}

void Layer::UpdateOutputGradient(float *buf) {
    if (m_OutputGrad == nullptr) return;

    auto curOutDim = GetCurrentOutputDimension();
    int outCnt = 1;
    for (auto d : curOutDim) outCnt *= d;

    for (int i = 0; i < outCnt; i++) {
        m_OutputGrad[i] += buf[i];
    }
}

Layer **Layer::GetInLayers() {
    if (m_inLayers.empty()) return nullptr;
    return &m_inLayers[0];
}

size_t Layer::GetInLayersCount() {
    return m_inLayers.size();
}

float Layer::Optimize(int idx, float prev_param, float new_grad) {
    return m_pOptimizer->Optimize(m_nOptStartIndex + idx, prev_param, new_grad);
}

void Layer::AddInLayer(Layer *pLayer) {
    if (pLayer == nullptr) {
        throw std::exception("Layer::AddInLayer: pLayer should not be NULL");
    }
    m_inLayers.push_back(pLayer);
}

void Layer::BroadcastInputGradient(float *input_grad) {
    for (Layer *pInput : m_inLayers) {
        pInput->UpdateOutputGradient(input_grad);
    }
}
