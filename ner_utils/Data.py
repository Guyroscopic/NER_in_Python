import pandas as pd

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import text_to_word_sequence, Tokenizer

class Data:
    
    def __init__(self, filename, encoding):
        self.filename         = filename
        self.encoding         = encoding
        self.dataset          = self.load_dataset()
        
    def load_dataset(self): return pd.read_csv(self.filename, encoding=self.encoding).fillna(method='ffill')

    def _tokenize_data(self, vocab, lower=True, char_level=False):
        t = Tokenizer(lower=lower, oov_token='UNK', char_level=char_level)
        t.fit_on_texts(list(vocab))

        return t

    def get_tokens(self, words, sentences):

    	char_to_idx = self._tokenize_data(words, lower=False, char_level=True).word_index
    	word_to_idx = self._tokenize_data(sentences, lower=True).word_index

    	return (
				char_to_idx,
    		    word_to_idx, 
    		    { idx : char for char, idx in char_to_idx.items()}, 
    		   	{ idx : word for word, idx in word_to_idx.items()}
    		)

    def get_tokenized_sequences(self, words, sentences, max_word_len, max_sentence_len):

    	char_to_idx, word_to_idx, idx_to_char, idx_to_word = self.get_tokens(words, sentences)

    	sentence_sequences = [[word_to_idx[w.lower()] for w in s] for s in sentences]
    	sentence_sequences = pad_sequences(sequences=sentence_sequences, maxlen=max_sentence_len, value=0, padding='post', truncating='post')

    	word_sequences     =  [pad_sequences([[char_to_idx[c] for c in w] for w in s], maxlen=max_word_len, value=0, padding='post', truncating='post') for s in sentences]

    	return word_sequences, sentence_sequences
    

    # def display_data(self, n):
    #     return self.dataset.head(n)
      
  
    # def data_info(self):
    #     n_sentences = len(self.data)
        
    #     words     = list(set(self.data['Word'].values))
    #     n_words   = len(words)
        
    #     tags      = list(set(self.data['Tag'].values))
    #     n_tags    = len(tags)
        
    #     print(f"Total Sentences : {n_sentences}\nTotal Number of Words : {n_words}\nTotal Number of Tags : {n_tags}")
           
        