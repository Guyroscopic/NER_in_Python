import pandas as pd
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer

class Data:
    
    def __init__(self, filename, encoding):
        self.filename         = filename
        self.encoding         = encoding
        self.dataset          = self.load_dataset()
        
    def load_dataset(self):
        data = pd.read_csv(self.filename, encoding=self.encoding).fillna(method='ffill')
        return data

    def tokenize_data(self, vocab, lower=True, char_level=False):
        t = Tokenizer(lower=lower, oov_token='UNK', char_level=char_level)
        t.fit_on_texts(list(vocab))

        return t

    
    # def display_data(self, n):
    #     return self.dataset.head(n)
      
  
    # def data_info(self):
    #     n_sentences = len(self.data)
        
    #     words     = list(set(self.data['Word'].values))
    #     n_words   = len(words)
        
    #     tags      = list(set(self.data['Tag'].values))
    #     n_tags    = len(tags)
        
    #     print(f"Total Sentences : {n_sentences}\nTotal Number of Words : {n_words}\nTotal Number of Tags : {n_tags}")
           
        