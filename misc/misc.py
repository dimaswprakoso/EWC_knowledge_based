# from nltk.corpus import brown
# import math
#
# brown_freqs = dict()
#
# def info_content(lookup_word):
#     """
#     Uses the Brown corpus available in NLTK to calculate a Laplace
#     smoothed frequency distribution of words, then uses this information
#     to compute the information content of the lookup_word.
#     """
#     N = 0
#     if N == 0:
#         # poor man's lazy evaluation
#         for sent in brown.sents():
#             for word in sent:
#                 word = word.lower()
#                 if word not in brown_freqs:
#                     brown_freqs[word] = 0
#                 brown_freqs[word] = brown_freqs[word] + 1
#                 N = N + 1
#     lookup_word = lookup_word.lower()
#     n = 0 if lookup_word not in brown_freqs else brown_freqs[lookup_word]
#     return 1.0 - (math.log(n + 1) / math.log(N + 1))
#
#
# ic_word = 'oxid'
# print('information content value of %s is %.4f' % (ic_word, info_content(ic_word)))

# from nltk.corpus import wordnet as wn
# from nltk.corpus import brown
#
# brown_ic = wn.ic(brown, False, 0.0)
#
# sysnset1 = wn.synset('iron.n.01')
# sysnset2 = wn.synset('soft.a.01')
#
# # sim = dog.lin_similarity(cat, brown_ic)
# sim = wn.lin_similarity(sysnset1, sysnset2, brown_ic)
# print(sim)

# from difflib import SequenceMatcher
# word1 = 'alcaly'
# word2 = 'clay'
# print(SequenceMatcher(None, word1, word2).ratio())


# from similarity.ngram import NGram
#
# twogram = NGram(1)
# print(twogram.distance('metal', 'metallic'))

#

# ==================================================
# from __future__ import print_function
# from nltk.metrics.distance import *
# from difflib import SequenceMatcher
# import time
#
# start = time.time()
# i = 0
# while i < 100000:
#     sim = jaro_winkler_similarity('alcaly','clay')
#     i = i + 1
# print(jaro_winkler_similarity('alcaly','clay'))
# print(time.time() - start)
# print()
#
#
# start = time.time()
# i = 0
# while i < 100000:
#     sim = 1 - jaccard_distance(set('alcaly'),set('clay'))
#     i = i + 1
# print(1 - jaccard_distance(set('alcaly'),set('clay')))
# print(time.time() - start)
# print()
#
#
# start = time.time()
# i = 0
# while i < 100000:
#     sim = SequenceMatcher(None, 'alcaly', 'clay').ratio()
#     i = i + 1
# print(SequenceMatcher(None, 'alcaly', 'clay').ratio())
# print(time.time() - start)
# print()
#
#
# from similarity.jarowinkler import JaroWinkler
#
# jarowinkler = JaroWinkler()
# start = time.time()
# i = 0
# while i < 100000:
#     sim = jarowinkler.similarity('alcaly', 'clay')
#     i = i + 1
# print(jarowinkler.similarity('alcaly', 'clay'))
# print(time.time() - start)
# print()

# ==================================================


# list1 = ['aaa', 'bbb', 'ccc']
# list2 = ['ddd', 'eee', 'fff']
# join = list1 + list2
# list_join = []
# list_join.append('111')
# print()

# x=['aaaaa','ddddd','bbbbb','ccccc','xxxx']
# print(x)
# print(set(x))
#
# import ari as ar
# print(ar.jumlah(1,2))
# import word_sim as ws
# from nltk.corpus import wordnet as wn
# synset1 = wn.synset('cat.n.01')
# synset2 = wn.synset('dog.n.01')
# print(ws.li_similarity(synset1, synset2))

# import pandas as pd
# from collections import OrderedDict
#
# dict1 = {'tp':[1], 'fp':[2], 'tn':[1], 'fn':[1]}
# df = pd.DataFrame.from_dict(dict1)
# print(df)

# import result_log as logging
# result = {'no_items':356,
#        'no_rec':346, 'no_labeled':312, 'tp':74, 'fp':282, 'tn':10, 'fn':10, 'precision':0.20786516853932585,
#        'recall':0.23717948717948717, 'accuracy':0.23595505617977527, 'f1':0.2215568862275449}
# log = {'test_type':'wup-lemma', 'pos':'n', 'wordsim_th':0.8, 'top_n':10, 'duration':808.5}
# log.update(result)
# logging.log_result(log)

import word_sim as ws
# from nltk.corpus import wordnet as wn
# # from word_sim import li_similarity
# import word_sim as ws
#
# sysnset1 = wn.synset('dog.n.01')
# sysnset2 = wn.synset('cat.n.01')
# sim = ws.li_similarity(sysnset1, sysnset2)
# print(sim)

# PHI = 0.1
# for i in range(5,16,5):
#     PHI = i
#     print(PHI)


# from scipy import spatial
# sim = 1 - spatial.distance.cosine([0.3,0.6,0.2,0.5,1,0.8], [0.1,0.7,1,0.5,1,0.7])
# print("similarity: %s" % sim)


# import mysql.connector
#
# db_user = 'root'
# db_database = 'sharebox'
# language = 'EN'
# cnx = mysql.connector.connect(user=db_user, database=db_database)
# cursor = cnx.cursor(dictionary=True)
# sql = "delete from log_result where tes"

# from nltk.corpus import wordnet as wn
# from nltk.corpus import brown
# from nltk.corpus import wordnet_ic
# import time
#
# sysnset1 = wn.synset('glass.n.01')
# sysnset2 = wn.synset('ammonium.n.01')
#
# start = time.time()
# brown_ic = wn.ic(brown, False, 0.0)
# sim = wn.lin_similarity(sysnset1, sysnset2, brown_ic)
# print(time.time() - start)
#
# start = time.time()
# brown_ic = wordnet_ic.ic('ic-brown.dat')
# sim = wn.lin_similarity(sysnset1, sysnset2, brown_ic)
# print(time.time() - start)




# from nltk.corpus import wordnet as wn
# from nltk.corpus import brown
# from nltk.corpus import wordnet_ic
# brown_ic = wordnet_ic.ic('ic-brown.dat')
# word_1 = 'available'
# word_2 = 'economy'
#
# synsets_1 = wn.synsets(word_1)
# synsets_2 = wn.synsets(word_2)
# if len(synsets_1) == 0 or len(synsets_2) == 0:
#     best_pair = None, None
# else:
#     max_sim = -1.0
#     best_pair = None, None
#     for synset_1 in synsets_1:
#         for synset_2 in synsets_2:
#
#             # ignore if both words are from different POS or not Noun type
#             if synset_1._pos != synset_2._pos or synset_1._pos == 's' or synset_2._pos == 's' or synset_1._pos == 'a' or synset_2._pos == 'a' or synset_1._pos == 'r' or synset_2._pos == 'r':  # for Lin_similarity
#                 sim = 0
#             else:
#                 sim = wn.lin_similarity(synset_1, synset_2, brown_ic)
#             if sim == None:
#                 sim = 0
#             if sim > max_sim:
#                 max_sim = sim
#                 best_pair = synset_1, synset_2, max_sim
# # ver=wn.get_version()
# print(best_pair)




# from nltk.corpus import brown
# import math
# from nltk.corpus import wordnet_ic
#
#
# brown_ic = wordnet_ic.ic('ic-brown.dat')
# brown_freqs = dict()
# ic = brown_ic('dog')

# #
#
# from scipy import spatial
#
# def cos_similarity(item_vec1, item_vec2):
#     sim = 1 - spatial.distance.cosine(item_vec1, item_vec2)  # cosine similarity
#
#     return sim
#
# print(cos_similarity([1,1,1,0,0,0,0],[0,0,0,1,1,0,0]))
# print(cos_similarity([1,1,1,0,0],[0,0,0,1,1]))

# PHI = 1.0
# if max_sim > PHI:
#     print(max_sim)
# else:
#     0.0

# sent_sim = 'croft'
# word_sim='path'
# base_word='lemma'
# pos = 'noun'
# db_table = "word_sim_%s_%s_%s_%s" % (sent_sim,word_sim,base_word,pos)
# sql_truncate = "TRUNCATE TABLE %s" % db_table
#
# sql = """INSERT INTO """ + db_table + """(word1, word2, similarity) VALUES (%s, %s, %s)"""
#
# # print()
#
# def f():
#     print(s)
# s = "I love Paris in the summer!"
# f()

# top_n_list = [5,10,15]
# for top_n in top_n_list:
#     print(top_n)

top_n = 5
for i in range(top_n):
    print(i)