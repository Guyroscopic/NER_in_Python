import pickle
import string
import sys
import numpy as np

def get_char_feature(char):
    return np.array([
        char.islower(),                 #Char is Lower Case
        char.isupper(),                 #Char is Upper Case
        char in string.punctuation,     #Char is Punctuation
        char.isnumeric(),               #Char is Numeric
        not( char.islower() or char.isupper() or char in string.punctuation or char.isnumeric())   #Char is None of the Above
    ], dtype=np.int32)

def main(add_char_features=False):

    np.random.seed(10)      #to reproduce results

    embedding_dim   = 30
    limit           = np.sqrt(3 / embedding_dim)    #According to 4 - NERresearch paper

    with open('unique_chars.pkl', 'rb') as f:
        chars = pickle.load(f)

    char_embeddings = dict()
    for c in chars:
        embedding           = np.random.uniform(-limit, limit, embedding_dim)
        char_embeddings[c]  = np.concatenate((embedding, get_char_feature(c))) if add_char_features else embedding        

    file_name = 'char_embeddings_with_features.pkl' if add_char_features else 'char_embeddings.pkl'
 
    with open(file_name, 'wb') as f:
        pickle.dump(char_embeddings, f) 

    print(char_embeddings)

if __name__ == '__main__':

    try:                sys.argv[1]; add_char_features = True        
    except IndexError:  add_char_features = False


    print(add_char_features)
    main(add_char_features=add_char_features)


