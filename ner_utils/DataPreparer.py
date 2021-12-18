import pandas as pd
import numpy  as np

from tensorflow.keras.preprocessing.sequence    import pad_sequences
from tensorflow.keras.preprocessing.text        import text_to_word_sequence, Tokenizer


class DataPreparer:

    def __init__(self, filename, encoding):
        """
        Constructor
        """

        self.dataset = self._load_dataset(filename, encoding)

    ###-----START PUBLIC METHODS-----###

    def get_tokenized_sentences(self, max_word_len):
        """
        Public method for ...

        @return
        """

        word_to_idx          = self._get_word_tokens()
        word_to_idx['<PAD>'] = 0

        temp_dataset               = self.dataset.copy()
        temp_dataset['word_token'] = temp_dataset.apply(lambda row: word_to_idx[row['Word'].lower()], axis=1)
        tokenized_sentences        = temp_dataset.groupby(['Sentence #'])['word_token'].apply(np.array)
        padded_tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=max_word_len, value=0, padding='post', truncating='post') 

        return padded_tokenized_sentences, word_to_idx

    
    def get_tokenized_words(self, max_word_len):
        """
        Public method for ...

        @return
        """
        pass
        
    ###-----END PUBLIC METHODS-----###


    ###-----START PRIVATE METHODS-----###

    def _get_char_tokens(self):
        """
        Private method to generate character tokens.

        @return
        A tuple of dictionaries, first maps chars words to integer tokens, 
        second maps integer tokens to chars
        """

        words     = self.dataset['Word'].values

        tokenizer = Tokenizer(lower=False, oov_token='<UNK>', char_level=True)
        tokenizer.fit_on_texts(list(words))

        return tokenizer.word_index


    def _get_word_tokens(self):
        """
        Private method to generate word tokens.

        @return
        A tuple of dictionaries, first maps words to integer tokens, 
        second maps integer tokens to words
        """

        sentences = self.dataset.groupby(['Sentence #'])['Word'].transform(lambda word : ' '.join(word)).drop_duplicates()

        tokenizer = Tokenizer(filters="", lower=True, oov_token='<UNK>', char_level=False)
        tokenizer.fit_on_texts(list(sentences))

        return tokenizer.word_index


    def _load_dataset(self, filename, encoding): 
        """
        Private method for loading dataset from .csv file
        """

        return pd.read_csv(filename, encoding=encoding).fillna(method='ffill')

    ###-----END PRIVATE METHODS-----###
