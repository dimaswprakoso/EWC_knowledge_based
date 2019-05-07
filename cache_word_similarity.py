import mysql.connector
import datetime
import time
import math
import word_sim as ws #contain Li's word similarity

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import brown
from nltk.corpus import wordnet_ic



# ----------------------------------------------------------------------------#
# Configuration
# ----------------------------------------------------------------------------#
starttime = datetime.datetime.now()
print("start:%s" % starttime)


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
sql = "TRUNCATE TABLE word_sim_cache"
# sql_ic = "TRUNCATE TABLE ic_cache"
try:
    cursor.execute(sql)
    # cursor.execute(sql_ic)

except mysql.connector.Error as e:
    print("x Failed inserting data: {}\n".format(e))


# ----------------------------------------------------------------------------#
# preprocessing
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
    term_list = ['waste', 'wastes', 'scrap', 'scraps' 'process', 'consultancy', 'advice', 'training', 'service',
                 'managing', 'management', 'recycling', 'recycle', 'industry', 'industrial', 'material', 'quantity',
                 'support', 'residue', 'organic', 'remainder']
    words = [w for w in words if not w in term_list]  # for each word check if

    # 5. Find Stem # Porter Stemmer

    # ps = PorterStemmer()
    # stemmed_words = []
    # for w in words:
    #     stemmed_words.append(ps.stem(w))
    # data = stemmed_words

    lm = WordNetLemmatizer()
    lemmatized_words = []
    for w in words:
        lemmatized_words.append(lm.lemmatize(w))
    data = lemmatized_words

    return data


# ----------------------------------------------------------------------------#
# Similarity Functions
# ----------------------------------------------------------------------------#
def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    return synset_pair[2]


def get_best_synset_pair(word_1, word_2):
    """
    Choose the pair with highest path similarity among all pairs.
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

                # ignore if both words are from different POS or not Noun type
                # if synset_1._pos != synset_2._pos or synset_1._pos == 's' or synset_2._pos == 's': # for Lin_similarity
                if (synset_1._pos != synset_2._pos or synset_1._pos != 'n' or synset_2._pos != 'n'):
                    sim = 0
                else:
                    # sim = wn.lin_similarity(synset_1, synset_2, brown_ic)
                    # sim = wn.wup_similarity(synset_1, synset_2)
                    sim = ws.li_similarity(synset_1, synset_2)
                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2, max_sim
        return best_pair


# def info_content(lookup_word): # for Li's sentence similarity
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

# ----------------------------------------------------------------------------#
# Main code
# ----------------------------------------------------------------------------#
# --------------- Config -------------------- #
# brown_freqs = dict()
# N = 0
# brown_ic =  wordnet_ic.ic('ic-brown.dat')

ewc_words = {}  # bag of words from the ewc description
item_words = {}  # bag of words from the item description
all_unique_words = []
# --------------- End Config ---------------- #

# Prepare the item_desc
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

# build word similarity and information content cache
print("unique words: {}".format(len(all_unique_words)))

rows = []
# ic_rows = []
for first_word in all_unique_words:
    # ic_row = (first_word, info_content(first_word))
    # ic_rows.append(ic_row)
    for second_word in all_unique_words:
        if first_word == second_word:
            word_sim = 1
        else:
            word_sim = word_similarity(first_word, second_word)

        row = (first_word, second_word, word_sim)
        rows.append(row)

sql = """INSERT INTO word_sim_cache(word1, word2, similarity) VALUES (%s, %s, %s)"""
# sql_ic = """INSERT INTO ic_cache(word1, ic) VALUES (%s, %s)"""
try:
    cursor.executemany(sql, rows)
    # cursor.executemany(sql_ic, ic_rows)
    cnx.commit()
except mysql.connector.Error as e:
    print("x Failed inserting data: {}\n".format(e))

endtime = datetime.datetime.now()
print("start  :%s" % starttime)
print("end    :%s" % endtime)
print("elapsed:%s" % (endtime - starttime))
