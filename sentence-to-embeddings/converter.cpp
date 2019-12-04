
#include "pch.h"
#include "converter.h"


converter::converter()
{
    m_W = nullptr;
    m_Wbuf = nullptr;
}

converter::~converter()
{
    if (m_W != nullptr) {
        delete[] m_W;
    }
    if (m_Wbuf != nullptr) {
        delete[] m_Wbuf;
    }
}

void converter::load_embedding_word2vec(const wchar_t *Path)
{
    std::ifstream fp(Path, std::ios::binary);
    char buf[4000];
    size_t len;

    auto read = [&](char delim) -> void {
        len = 0;
        do {
            fp.read(&buf[len++], 1);
        } while (buf[len - 1] != delim);
        buf[len-- - 1] = '\0';
    };

    /* read header */
    read('\n');
    sscanf_s(buf, "%d %d", &m_V, &m_D);

    /* allocate memory */
    m_Wbuf = new float[m_V * m_D];
    m_W = new float*[m_V];
    for (int i = 0; i < m_V; i++) {
        m_W[i] = m_Wbuf + (i*m_D);
    }

    /* read word vectors */
    for (int i = 0; i < m_V; i++) {
        read(' ');
        fp.read((char *)(m_W[i]), sizeof(float)*m_D);

        auto s = std::string(buf);
        m_dict[s] = i;
        m_words.push_back(s);
    }

    fp.close();
}

void converter::word_sim_k(std::string word, int k)
{
    auto it = m_dict.find(word);
    if (it == m_dict.end()) {
        printf("Word [%s] not found in dictionary\n", word.c_str());
        return;
    }
    int wid = it->second;

    /* iterate over all words */
    std::vector<std::pair<int, float>> scores;
    for (int wref = 0; wref < m_V; wref++) {
        if (wref == wid) continue;

        float sim = 0;
        float vec_sz = 0;
        float wid_sz = 0;
        for (int j = 0; j < m_D; j++) {
            sim += m_W[wid][j] * m_W[wref][j];
            vec_sz += m_W[wref][j] * m_W[wref][j];
            wid_sz += m_W[wid][j] * m_W[wid][j];
        }

        sim /= sqrt(vec_sz) * sqrt(wid_sz);
        scores.push_back(std::make_pair(wref, sim));
    }
    std::sort(scores.begin(), scores.end(), [](const std::pair<int, float> &a, const std::pair<int, float> &b) -> bool {
        return a.second > b.second;
    });

    /* print the top k words */
    printf("top %d words similar to %s:\n", k, word.c_str());
    for (int i = 0; i < k; i++) {
        printf("    %s: %1.8f\n", m_words[scores[i].first].c_str(), scores[i].second);
    }
    printf("\n");
}

void converter::convert_and_save(const wchar_t *SrcPath, const wchar_t *DstPath)
{
    bool skip_oov = true;

    std::ifstream fpi(SrcPath);
    std::ofstream fpo1(DstPath + std::wstring(L".stc"), std::ios::binary);
    std::ofstream fpo2(DstPath + std::wstring(L".vec"), std::ios::binary);
    std::ofstream fpo1t(DstPath + std::wstring(L".stc.txt"));
    std::ofstream fpo2t(DstPath + std::wstring(L".vec.txt"));

    char buf[10000];
    size_t len;

    auto read = [&](char delim) -> void {
        len = 0;
        do {
            fpi.read(&buf[len++], 1);
        } while (buf[len - 1] != delim && !fpi.eof());
        buf[len-- - 1] = '\0';
    };

    /* write header */
    int stc = 0;

    /* conversion */
    std::vector<int> stcs;
    std::map<std::string, int> new_dict;
    std::vector<float> embs;
    float *h = new float[m_D];
    while (!fpi.eof()) {
        read('\n');
        char *Context;
        char *start = strtok_s(buf, " ", &Context);
        std::vector<int> stc_line;
        while (start != nullptr) {
            int wid;
            auto it = new_dict.find(std::string(start));
            if (it != new_dict.end()) {
                wid = it->second;
            }
            else {
                bool oov = !get_word_vector(start, h);
                if (oov) {
                    if (skip_oov) {
                        start = strtok_s(nullptr, " ", &Context);
                        continue;
                    }

                    /* random init */
                    for (int j = 0; j < m_D; j++) {
                        h[j] = -0.5 + rand() / (double)RAND_MAX;
                    }
                }
                wid = (int)new_dict.size();
                new_dict[start] = wid;
                for (int j = 0; j < m_D; j++) embs.push_back(h[j]);
            }
            stc_line.push_back(wid);
            start = strtok_s(nullptr, " ", &Context);
        }
        stcs.push_back((int)stc_line.size());
        for (int wid : stc_line) stcs.push_back(wid);
        stc++;
    }

    /* write to file */
    fpo1.write((const char *)&stc, 4); // number of sentences
    fpo1.write((const char *)&stcs[0], 4 * stcs.size());
    fpo1.close();

    int words = (int)new_dict.size();
    fpo2.write((const char *)&words, 4); // number of words
    fpo2.write((const char *)&embs[0], 4 * embs.size());
    fpo2.close();

    fpo1t << stc;
    int count_remaining = 0;
    for (int wid : stcs) {
        if (count_remaining == 0) {
            count_remaining = wid;
            fpo1t << "\n";
        }
        else {
            fpo1t << wid << " ";
            count_remaining--;
        }
    }
    fpo1t.close();

    fpo2t << words << " " << m_D << "\n";
    int cur_count = 0;
    for (float c : embs) {
        cur_count++;
        if (cur_count == m_D) {
            fpo2t << c << "\n";
            cur_count = 0;
        }
        else {
            fpo2t << c << " ";
        }
    }
    fpo2t.close();

    /* cleanup */
    delete[] h;

#if 0
    /* conversion (old) */
    char *Context;
    int oov = 0;
    int inv = 0;
    std::string st;
    while (!fpi.eof()) {
        tmp = 0;
        fpo.write((const char *)&tmp, 4); // for word count
        st.clear();

        read('\n');
        char *start = strtok_s(buf, " ", &Context);
        memset(havg, 0, sizeof(float)*m_D);
        while (start != nullptr) {
            bool success = get_word_vector(start, h);
            if (success) {
                fpo.write((const char *)h, sizeof(float)*m_D);
                tmp++;
                inv++;
                st += start;
                for (int j = 0; j < m_D; j++) havg[j] += h[j];
            }
            else {
                oov++;
                st += '[';
                st += start;
                st += ']';
            }
            st += ' ';
            start = strtok_s(nullptr, " ", &Context);
        }
        st[st.length() - 1] = '\n';
        fpo2.write(st.c_str(), st.length());

        fpo.seekp(-tmp * sizeof(float)*m_D - 4, std::ios::cur);
        fpo.write((const char *)&tmp, 4);
        fpo.seekp(tmp * sizeof(float)*m_D, std::ios::cur);

        //for (int j = 0; j < m_D; j++) havg[j] /= inv;
        fpo3.write((const char *)havg, sizeof(float)*m_D);

        stc++;
    }

    fpo.seekp(0, std::ios::beg);
    fpo.write((const char *)&stc, 4);
    fpo.close();
    fpo2.close();
    fpo3.close();

    delete[] h;
    delete[] havg;

    printf("oov = %d, inv = %d\n", oov, inv);
#endif
}

bool converter::get_word_vector(const char *String, float *h)
{
    auto it = m_dict.find(std::string(String));
    if (it == m_dict.end()) return false;

    memcpy_s(h, sizeof(float)*m_D, m_W[it->second], sizeof(float)*m_D);
    return true;
}
