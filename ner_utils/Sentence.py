#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

# In[2]: 

class Sentence:
    
    
    def __init__(self, data):
        self.data      = data
        _, self.sentences = self.get_words_and_sentences()
        self.tags      = self.get_unique_tags()
        self.chars     = self.get_unique_chars()
    

    def get_words_and_sentences(self):

      # removing unwanted char from text
      self.data['Word'] = self.data['Word'].apply(lambda s: s.replace('\xa0', ''))
      
      return (
        self.data['Word'].values,
        self.data.groupby('Sentence #').apply(lambda s: [w for w in s["Word"].values.tolist()])
        )


    def get_sentence_info(self):


      ''' 
      Output: Max sentence length, Min sentence length, Mean sentence length, Standard deviation of sentence length

      '''
      sentence_lengths = [len(s) for s in self.sentences]
      
      return (
         np.max(sentence_lengths),     # maximum sentence length
         np.min(sentence_lengths),     # maximum sentence length
         np.mean(sentence_lengths),    # mean sentence length
         np.std(sentence_lengths)      # standard deviatio of sentence length
        )
        

    def get_word_info(self): 
      word_lengths = [len(w) for s in self.sentences for w in s] #word length
      
      return (    
         np.max(word_lengths), #maximum word length
         np.min(word_lengths), #minimum word length
         np.mean(word_lengths), #mean word length
         np.std(word_lengths) #standard deviatio of word length
      )         
       

    def get_labels(self):

      one_hot = pd.concat([self.data, pd.get_dummies(self.data['Tag'])], axis=1)
      return one_hot.groupby('Sentence #').apply(lambda s : s[[col for col in s.columns[4:]]].values.tolist())

    
    def get_unique_chars(self): return list(set([char for w in self.data["Word"].values.tolist() for char in w]))

    def get_unique_tags(self) : return list(set([tag for tag in self.data["Tag"].values.tolist()]))
    




# In[ ]:




