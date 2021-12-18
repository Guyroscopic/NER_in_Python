import pickle
import string
import sys
import numpy as np

def get_char_feature(char):
    """
    Takes in a character and returns a numpy array of pre-defined char features
    """

    return np.array([
        char.islower(),                 #Char is Lower Case
        char.isupper(),                 #Char is Upper Case
        char in string.punctuation,     #Char is Punctuation
        char.isnumeric(),               #Char is Numeric
        not(char.islower() or char.isupper() or char in string.punctuation or char.isnumeric())   #Char is None of the Above
    ], dtype=np.int32)

def write_to_txt_file(dict, file_name):
    """
    Takes in a dictionary of embeddings and writes them to a .txt file with the same format as GloVe Embeddings
    """
    string = ""
    for char, embedding in dict.items():
        temp = " "
        temp += char + " " + temp.join(embedding.astype(np.str)) + '\n'        
        string += temp[1:]

    with open(file_name, 'w', encoding='latin') as f:
        f.write(string[:-1])


def main(add_char_features=False):

    np.random.seed(10)      #to reproduce results

    embedding_dim   = 30
    limit           = np.sqrt(3 / embedding_dim)    #According to 4 - NER research paper

    #####-----CHANGE THIS---------#####
    with open('unique_chars.pkl', 'rb') as f:
        chars = pickle.load(f)
    #####-----CHANGE THIS---------#####

    char_embeddings = dict()
    for c in chars:
        embedding           = np.random.uniform(-limit, limit, embedding_dim)
        char_embeddings[c]  = np.concatenate((embedding, get_char_feature(c))) if add_char_features else embedding        

    file_name = 'char_embeddings_with_features.txt' if add_char_features else 'char_embeddings.txt'
    
    # with open(file_name, 'wb') as f:
    #     pickle.dump(char_embeddings, f) 

    write_to_txt_file(char_embeddings, 'char embeddings/'+file_name)


if __name__ == '__main__':

    try:                sys.argv[1]; add_char_features = True if sys.argv[1] == 'true' else False   
    except IndexError:               add_char_features = False

    main(add_char_features=add_char_features)


