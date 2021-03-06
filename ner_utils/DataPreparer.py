from numpy.lib.arraysetops import union1d
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

    def get_tokenized_sentences(self, max_sentence_len):
        """
        Public method for ...

        @return
        """

        pad_value = 0

        word_to_idx          = self._get_word_tokens()
        word_to_idx['<PAD>'] = pad_value

        temp_dataset               = self.dataset.copy()
        temp_dataset['word_token'] = temp_dataset['Word'].str.lower().map(word_to_idx)           
        tokenized_sentences        = temp_dataset.groupby(['Sentence #'])['word_token'].apply(np.array)
        padded_tokenized_sentences = pad_sequences(tokenized_sentences, maxlen=max_sentence_len, value=pad_value, padding='post', truncating='post') 

        return padded_tokenized_sentences, word_to_idx

    
    def get_tokenized_words(self, max_word_len, max_sentence_len):
        """
        Public method for ...

        @return
        """

        pad_value = 0

        char_to_idx          = self._get_char_tokens()
        char_to_idx['<PAD>'] = pad_value

        temp_dataset                = self.dataset.copy()
        temp_dataset['char_tokens'] = self.dataset['Word'].str.split("").str[1:-1]
        temp_dataset['char_tokens'] = temp_dataset['char_tokens'].apply(lambda x: [char_to_idx[i] for i in x])
        padded_tokenized_words      = temp_dataset.groupby(['Sentence #'])['char_tokens'].apply(np.array).apply(pad_sequences, maxlen=max_word_len, value=pad_value , padding='post', truncating='post')
        padded_tokenized_words      = pad_sequences(padded_tokenized_words, maxlen=max_sentence_len, value=[pad_value]*max_word_len , padding='post', truncating='post')

        return padded_tokenized_words, char_to_idx


    def get_tags(self):
        """
        Public method for ...

        @return
        """

        unique_tags = self._get_unique_tags()
        tag_to_idx  = { tag: idx for idx, tag in enumerate(unique_tags) }

        temp_dataset = self.dataset.copy()


        temp_dataset['tag_token'] = temp_dataset['Tag'].str.lower().map(tag_to_idx)
        
                   
        tokenized_sentences        = temp_dataset.groupby(['Sentence #'])['word_token'].apply(np.array)





       
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


    def _get_unique_tags(self):

        return self.dataset['Tag'].unique()

    ###-----END PRIVATE METHODS-----###
