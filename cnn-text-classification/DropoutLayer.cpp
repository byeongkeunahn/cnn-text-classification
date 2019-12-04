
#include "pch.h"
#include "DropoutLayer.h"
#include "Random.h"


DropoutLayer::DropoutLayer(Layer *pInput, float keep_prob)
{
    if (keep_prob < 0.01 || keep_prob > 1.0) {
        throw std::exception("DropoutLayer: Please supply keep probability within [0.01, 1.00]");
    }
    m_keepProb = keep_prob;

    auto indim = pInput->GetOutputDimension();
    if (indim.size() != 1 || indim[0] <= 0) {
        throw std::exception("DropoutLayer: Input layer has wrong dimensions");
    }
    m_dim = indim[0];
    m_Output = new float[m_dim];
    m_OutputGrad = new float[m_dim];
    m_InputGrad = new float[m_dim];
    m_fKeepMap = new float[m_dim];

    AddInLayer(pInput);
}

DropoutLayer::~DropoutLayer()
{
    delete[] m_Output;
    delete[] m_OutputGrad;
    delete[] m_InputGrad;
    delete[] m_fKeepMap;
}

std::vector<int> DropoutLayer::GetOutputDimension() {
    return std::vector<int>({ m_dim });
}

int DropoutLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void DropoutLayer::ForwardProp() {
    Layer *pInputLayer = m_inLayers[0]; // there is only one such layer
    float *input = (float *)pInputLayer->GetOutput();

    /* generate the keep map */
    if (m_isTesting) {
        for (int i = 0; i < m_dim; i++) {
            m_fKeepMap[i] = 1;
        }
    }
    else {
        for (int i = 0; i < m_dim; i++) {
            bool fKeep = Random::Uniform() < m_keepProb;
            m_fKeepMap[i] = (fKeep) ? 1 : 0;
        }
    }

    /* forward propagation */
    for (int i = 0; i < m_dim; i++) {
        m_Output[i] = input[i] * m_fKeepMap[i];
    }

    memset(m_OutputGrad, 0, sizeof(float)*m_dim);
}

void DropoutLayer::BackwardProp() {
    /* compute the gradients */
    for (int i = 0; i < m_dim; i++) {
        m_InputGrad[i] = m_OutputGrad[i] * m_fKeepMap[i];
    }

    /* broadcasting */
    BroadcastInputGradient(m_InputGrad);
}

float* DropoutLayer::GetOutput() {
    return m_Output;
}

void DropoutLayer::UpdateParams() {
    /* a dropout layer does not have any associated parameters */
}
