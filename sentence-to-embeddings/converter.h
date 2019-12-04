
#pragma once


class converter
{
public:
    converter();
    ~converter();

public:
    void load_embedding_word2vec(const wchar_t *Path);
    void word_sim_k(std::string word, int k);
    void convert_and_save(const wchar_t *SrcPath, const wchar_t *DstPath);

private:
    bool get_word_vector(const char *String, float *h); // returns -1 if not available

private:
    float **m_W, *m_Wbuf;
    int m_V, m_D;
    std::map<std::string, int> m_dict;
    std::vector<std::string> m_words;
};

