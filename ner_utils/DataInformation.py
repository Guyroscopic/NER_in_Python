import pandas as pd


class DataInformation:
    """
    Class to calculate sentence and word length information
    """

    def __init__(self, filename, encoding):
        """
        Constructor

        @param
        filename: Path to the csv file
        encoding: Encoding of the csv file
        """

        self.dataset  = self._load_dataset(filename, encoding)

    
    def get_word_length_info(self):
        """
        Public method to calculate word length information from loaded dataset

        @return
        A dictionary of the max, min, average and standard deviation of word length
        """

        temp_dataset             = self.dataset.copy()
        temp_dataset["word_len"] = temp_dataset["Word"].str.len()

        return {    
            "max": temp_dataset["word_len"].max(),
            "min": temp_dataset["word_len"].min(),
            "avg": temp_dataset["word_len"].mean(),
            "std": temp_dataset["word_len"].std()
        }


    def get_sentence_length_info(self):
        """
        Public Method to calculate sentence length information from loaded dataset
        """

        temp_dataset  = self.dataset.copy()
        sentence_size = temp_dataset.groupby(['Sentence #']).size()

        return {    
            "max": sentence_size.max(),
            "min": sentence_size.min(),
            "avg": sentence_size.mean(),
            "std": sentence_size.std()
        }


    def _load_dataset(self, filename, encoding): 
        """
        Private method for loading dataset from .csv file
        """
        return pd.read_csv(filename, encoding=encoding).fillna(method='ffill')

    



