# import numpy as np
from nltk.corpus import wordnet as wn
#
# a = np.array([1, 2])
# a.tolist()
# print(a)

sysnset_1 = wn.synset('cat.n.01')
sysnset_2 = wn.synset('dog.n.01')
sim = wn.wup_similarity(sysnset_1,sysnset_2, simulate_root=False)
print(sim)