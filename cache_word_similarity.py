import mysql.connector
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet as wn
# from nltk.corpus import brown
from scipy import spatial
from collections import OrderedDict
from operator import itemgetter
import math
import numpy as np
import sys
import time

# ----------------------------------------------------------------------------#
# Configuration
# ----------------------------------------------------------------------------#
start = time.time()

db_user = 'root'
db_database = 'sharebox'
language = 'EN'

# ----------------------------------------------------------------------------#
# 0. Initialize
# ----------------------------------------------------------------------------#
# Mysql Connection
cnx = mysql.connector.connect(user=db_user, database=db_database)
cursor = cnx.cursor(dictionary=True)

# ----------------------------------------------------------------------------#
# 1. Get items from workshop
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `workshop_items2` WHERE language = '""" + language + """' AND type='Material' 
"""

# RTRIM(LTRIM(wastecode)) != '99 99 99' LIMIT 0,25

item_list = {}
try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        item_list[row['id']] = [row['Waste_description'], row['Wastecode']]

    print("Items rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

# ----------------------------------------------------------------------------#
# 2. Get items from ewc
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `ewc_level3`
"""

ewc_list = {}

try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        ewc_list[row['EWC_level3']] = [row['description'], row['id']]

    print("EWC rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

# ----------------------------------------------------------------------------#
# 3. Empty table word similarity
# ----------------------------------------------------------------------------#
sql = "TRUNCATE TABLE wup_word_sim"
try:
    cursor.execute(sql)
    cnx.commit()

except mysql.connector.Error as e:
    print("x Failed inserting data: {}\n".format(e))


# ----------------------------------------------------------------------------#
# Similarity Functions
# ----------------------------------------------------------------------------#
def NLP(data):
    # 0. Lowercase
    data = data.lower()

    # 1. Tokenize # word tokenize (removes also punctuation)
    tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
    words = tokenizer.tokenize(data)

    # 2. Remove short words
    words = [w for w in words if len(w) > 2]

    # 3. Remove Stopwords # load stopwords from enlgihs language
    stop_words = set(stopwords.words("english"))
    words = [w for w in words if not w in stop_words]  # for each word check if

    # 4 Remove common terminology in waste listings e.g. (waste)
    term_list = ['waste', 'process', 'consultancy', 'advice', 'training', 'service', 'managing', 'management',
                 'recycling', 'recycle', 'industry', 'industrial', 'material', 'quantity', 'support', 'residue',
                 'organic', 'remainder']
    words = [w for w in words if not w in term_list]  # for each word check if

    # 5. Find Stem # Porter Stemmer
    ps = PorterStemmer()
    stemmed_words = []
    for w in words:
        stemmed_words.append(ps.stem(w))

    data = stemmed_words
    return data


def find_unique_words(data, data2):
    # merge words from all items, then remove duplicates with set datatype
    all_unique_words = []
    all_unique_words = list(set(all_unique_words + data))

    for i, j in data2.items():
        all_unique_words = list(set(all_unique_words + j))

    return all_unique_words


def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    # return (length_dist(synset_pair[0], synset_pair[1]) *
    #         hierarchy_dist(synset_pair[0], synset_pair[1]))
    return synset_pair[2]


def get_best_synset_pair(word_1, word_2):
    """
    Choose the pair with highest path similarity among all pairs.
    Mimics pattern-seeking behavior of humans.
    """
    max_sim = -1.0
    synsets_1 = wn.synsets(word_1)
    synsets_2 = wn.synsets(word_2)
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        return None, None, 0.0
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:
                sim = wn.wup_similarity(synset_1, synset_2, simulate_root=False)
                # sim = wn.path_similarity(synset_1, synset_2)
                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2, max_sim
        return best_pair
        # return synsets_1[0], synsets_2[0]


# ----------------------------------------------------------------------------#
# Main code
# ----------------------------------------------------------------------------#
ewc_words = {}  # bag of words from the ewc description
item_words = {}  # bag of words from the item description
all_unique_words = []

for i, j in item_list.items():
    # clean EWC data
    item_ewc = j[1].strip()

    # delete * symbol from ewc code
    # translation_table = dict.fromkeys(map(ord, '*'), None)
    # item_ewc = item_ewc.translate(translation_table)
    item_list[i] = [j[0], item_ewc.strip()]

    desc = j[0].strip()

    # Generate word list for each waste description
    item_words[i] = NLP(desc)

    all_unique_words = list(set(all_unique_words + item_words[i]))

# prepare the ewc_desc
for k, l in ewc_list.items():
    ewc_words[k] = NLP(l[0])
    all_unique_words = list(set(all_unique_words + ewc_words[k]))

word_sim_dict = {}

for first_word in all_unique_words:
    for second_word in all_unique_words:
        if first_word == second_word:
            word_sim = 1
        else:
            word_sim = word_similarity(first_word, second_word)
        # word_sim_dict[first_word, second_word] = word_sim

        sql = 'INSERT INTO wup_word_sim(word1, word2, similarity) VALUES ("{}","{}",{})'.format (first_word,second_word, word_sim)
        try:
            cursor.execute(sql)
            cnx.commit()
        except mysql.connector.Error as e:
            print("x Failed inserting data: {}\n".format(e))
print("end")
