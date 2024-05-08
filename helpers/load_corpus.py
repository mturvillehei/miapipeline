from datasets import load_dataset
import os

### 

def load_Corpus():
    if os.path.exists("./Data/Corpus5"):
        return
    Corpus_en = load_dataset("wikipedia", "20220301.en", trust_remote_code=True)

    ### File location of the full corpus
    print(Corpus_en.cache_files)

    print(Corpus_en.keys())
    print(Corpus_en['train'][0])
    print(Corpus_en['train'].features)
    print(Corpus_en['train'].shape)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
    #### Random #####
    
    Corpus_shuffled = Corpus_en['train'].shuffle(seed=1010)
    Corpus500k_rand = Corpus_shuffled.select(range(5))
    print(Corpus500k_rand
          )
if __name__ == "__main__":
    ### Loading and shuffling the data.
    shuffled_data = load_Corpus()
    