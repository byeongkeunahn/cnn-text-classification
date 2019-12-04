
#include "pch.h"
#include "ConcatenationLayer.h"


ConcatenationLayer::ConcatenationLayer(Layer **pInputs, int Count)
{
    auto get_dim = [](Layer *pInput) -> int {
        auto indim = pInput->GetOutputDimension();
        if (indim.size() != 1 || indim[0] <= 0) {
            throw std::exception("ConcatenationLayer: Input layer has wrong dimensions");
        }
        return indim[0];
    };

    if (Count <= 0) {
        throw std::exception("ConcatenationLayer: Count <= 0 is invalid");
    }

    m_dims.resize(Count);
    for (int i = 0; i < Count; i++) {
        Layer *pL = pInputs[i];
        m_dims[i] = get_dim(pL);
        AddInLayer(pL);
    }

    m_dimTotal = 0;
    for (auto dim : m_dims) m_dimTotal += dim;

    m_Output = new float[m_dimTotal];
    m_OutputGrad = new float[m_dimTotal];
}

ConcatenationLayer::~ConcatenationLayer()
{
    delete[] m_OutputGrad;
}

std::vector<int> ConcatenationLayer::GetOutputDimension() {
    return std::vector<int>({ m_dimTotal });
}

int ConcatenationLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void ConcatenationLayer::ForwardProp() {
    /* concatenation by copy */
    int cur = 0;
    for (size_t i = 0; i < m_dims.size(); i++) {
        int dim = m_dims[i];
        memcpy_s(m_Output + cur, sizeof(float)*dim, m_inLayers[i]->GetOutput(), sizeof(float)*dim);
        cur += dim;
    }

    memset(m_OutputGrad, 0, sizeof(float)*m_dimTotal);
}

void ConcatenationLayer::BackwardProp() {
    /* compute the input gradient: nothing to do, since we're just changing the shape */

    /* broadcasting */
    int cur = 0;
    for (size_t i = 0; i < m_dims.size(); i++) {
        int dim = m_dims[i];
        m_inLayers[i]->UpdateOutputGradient(m_OutputGrad + cur);
        cur += dim;
    }
}

float* ConcatenationLayer::GetOutput() {
    return m_Output;
}

void ConcatenationLayer::UpdateParams() {
    /* a softmax layer does not have any associated parameters */
}
