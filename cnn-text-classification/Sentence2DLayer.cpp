
#include "pch.h"
#include "Sentence2DLayer.h"


Sentence2DLayer::Sentence2DLayer(EmbeddingLayer *pEmbeddings, Layer *pWordSequence, bool fTrainEmbeddings) {
    auto indim = pWordSequence->GetOutputDimension();
    if (indim.size() != 1 || indim[0] <= -2 || indim[0] == 0) {
        throw std::exception("Sentence2DLayer: Supplied WordSequence layer has an incompatible output dimension");
    }
    m_indim = indim[0];

    auto embed_dim = pEmbeddings->GetOutputDimension();
    m_WordCount = embed_dim[0];
    m_D = embed_dim[1];

    m_pEmbeddings = pEmbeddings;
    m_pWordSequence = pWordSequence;
    m_fTrainEmbeddings = fTrainEmbeddings;

    AddInLayer(pEmbeddings);
    AddInLayer(pWordSequence);
}

Sentence2DLayer::~Sentence2DLayer() {
}

std::vector<int> Sentence2DLayer::GetOutputDimension() {
    return std::vector<int>({ m_indim, m_D });
}

std::vector<int> Sentence2DLayer::GetCurrentOutputDimension() {
    int indim = m_pWordSequence->GetCurrentOutputDimension()[0];
    return std::vector<int>({ indim, m_D });
}

int Sentence2DLayer::GetNumberOfParams() {
    return 0; // indicates non-necessity of optimizer
}

void Sentence2DLayer::ForwardProp() {
    float *seqs = m_pWordSequence->GetOutput();
    float *embeds = m_pEmbeddings->GetOutput();
    int N = m_pWordSequence->GetCurrentOutputDimension()[0];

    if (m_OutputBuffer.size() < N*m_D) {
        m_OutputBuffer.resize(N*m_D);
        m_OutputGradBuffer.resize(N*m_D);
    }

    for (int i = 0; i < N; i++) {
        int wid = (int)seqs[i];
        if (wid >= 0 && wid < m_WordCount) {
            float *h = embeds + (wid * m_D);
            memcpy_s(&m_OutputBuffer[i*m_D], sizeof(float)*m_D, h, sizeof(float)*m_D);
        }
        else {
            memset(&m_OutputBuffer[i*m_D], 0, sizeof(float)*m_D);
        }
    }

    m_OutputGrad = &m_OutputGradBuffer[0];
    memset(m_OutputGrad, 0, sizeof(float)*N*m_D);
}

void Sentence2DLayer::BackwardProp() {
    if (!m_fTrainEmbeddings) return;

    /* broadcasting */
    float *seqs = m_pWordSequence->GetOutput();
    float *embeds = m_pEmbeddings->GetOutput();
    int N = m_pWordSequence->GetCurrentOutputDimension()[0];

    for (int i = 0; i < N; i++) {
        int wid = (int)seqs[i];
        if (wid >= 0 && wid < N) {
            m_pEmbeddings->UpdateOutputGradientSingleWord(wid, &m_OutputBuffer[i*m_D]);
        }
    }
}

float* Sentence2DLayer::GetOutput() {
    return &m_OutputBuffer[0];
}

void Sentence2DLayer::UpdateParams() {
    /* a sentence-2D layer (kind of proxy) does not have any associated parameters */
}
