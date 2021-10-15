import pickle
import numpy as np
np.random.seed(10)      #to reproduce results

embedding_dim   = 30
limit           = np.sqrt(3 / embedding_dim)

with open('unique_chars.pkl', 'rb') as f:
    chars = pickle.load(f)


char_embeddings = dict()
for c in chars:
    char_embeddings[c] = np.random.uniform(-limit, limit, embedding_dim)

with open('char_embeddings.pkl', 'wb') as f:
    pickle.dump(char_embeddings, f) 


