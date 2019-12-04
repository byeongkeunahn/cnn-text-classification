// sentence-to-embeddings.cpp : 이 파일에는 'main' 함수가 포함됩니다. 거기서 프로그램 실행이 시작되고 종료됩니다.
//

#include "pch.h"
#include "converter.h"


int main()
{
    std::cout << "Hello World!\n";

    converter conv;
    conv.load_embedding_word2vec(
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\cnn-text-classification-keras-master\GoogleNews-vectors-negative300.bin)"
    );
    conv.word_sim_k("microsoft", 20);
    conv.word_sim_k("two", 20);
    conv.word_sim_k("have", 20);
    conv.convert_and_save(
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\sarcasm-stc.txt)",
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\01_preprocess\sarcasm-stc.txt.embed)"
    );
    /*conv.convert_and_save(
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\cnn-text-classification-keras-master\rt-polarity.neg)",
        LR"(D:\Dev\School2019\ML-2019\NEW (Sarcasm Classification)\cnn-text-classification-keras-master\rt-polarity.neg.embed)"
    );*/
    return 0;
}
