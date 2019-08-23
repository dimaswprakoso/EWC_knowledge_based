import mysql.connector
import datetime
# import nltk
import word_sim as ws #contain Li's word similarity

from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk.corpus import wordnet_ic
from difflib import SequenceMatcher


# ----------------------------------------------------------------------------#
# Configuration
# ----------------------------------------------------------------------------#
starttime = datetime.datetime.now()
print("start:%s" % starttime)
brown_ic = wordnet_ic.ic('ic-brown.dat')

db_user = 'root'
db_database = 'sharebox'
language = 'EN'

# testing---------#
string_sim = 'croft'  # croft:if the word is not in wordnet then apply string sim, li:0
word_sim_algo = 'lin' # path, wup, lin, li
base_word = 'lemma' # raw, stem, lemma
pos = 'all' # noun, all
# ----------------#

db_table = "word_sim_%s_%s_%s_%s" % (string_sim,word_sim_algo,base_word,pos)


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
sql = "TRUNCATE TABLE %s" % db_table

try:
    cursor.execute(sql)

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
    term_list = ['waste', 'scrap', 'scraps', 'process', 'processes', 'processed', 'processing', 'unprocessed',
                 'consultancy', 'advice', 'training', 'service', 'managing', 'management', 'recycling', 'recycle',
                 'industry', 'industrial', 'material', 'materials', 'quantity', 'support', 'residue', 'organic',
                 'remainder', 'specific', 'particular', 'solution', 'solutions', 'substance', 'substances', 'product',
                 'production', 'use', 'used', 'unused', 'consumption', 'otherwise', 'specified', 'based', 'spent',
                 'hazardous', 'dangerous','containing','other']

    words = [w for w in words if not w in term_list]  # for each word check if
    data = words

    # 5. Find Stem # Porter Stemmer

    if base_word=='stem':
        ps = PorterStemmer()
        stemmed_words = []
        for w in words:
            stemmed_words.append(ps.stem(w))
        data = stemmed_words
    elif base_word=='lemma':
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

    # if zero synsets are found, assign string similarity
    if len(synsets_1) == 0 or len(synsets_2) == 0:
        if string_sim =='croft':
            # string_sim = 1 - nltk.jaccard_distance(set(word_1), set(word_2))
            str_sim = SequenceMatcher(None, word_1, word_2).ratio()
            return None, None, str_sim
        elif string_sim =='li':
            return None, None, 0.0
    else:
        max_sim = -1.0
        best_pair = None, None
        for synset_1 in synsets_1:
            for synset_2 in synsets_2:

                # ignore if both words are from different POS
                if (synset_1._pos != synset_2._pos):
                    sim = 0
                else: # same pos
                    if pos == 'noun': # pos noun
                        if synset_1._pos != 'n' or synset_2._pos != 'n':
                            sim = 0
                        else:
                            if word_sim_algo == 'wup':
                                sim = wn.wup_similarity(synset_1, synset_2)
                            elif word_sim_algo =='lin':
                                sim = wn.lin_similarity(synset_1, synset_2, brown_ic)
                            elif word_sim_algo == 'li':
                                sim = ws.li_similarity(synset_1, synset_2)
                            else:
                                sim = wn.path_similarity(synset_1, synset_2)
                    else: # pos all
                        if word_sim_algo == 'wup':
                            sim = wn.wup_similarity(synset_1, synset_2)
                        elif word_sim_algo == 'lin':
                            if (synset_1._pos=='v' or synset_1._pos=='n') and (synset_2._pos=='v' or synset_2._pos=='n'):
                                sim = wn.lin_similarity(synset_1, synset_2, brown_ic)
                            else:
                                sim=0
                        elif word_sim_algo == 'li':
                            sim = ws.li_similarity(synset_1, synset_2)
                        else:
                            sim = wn.path_similarity(synset_1, synset_2)

                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2, max_sim
        return best_pair


# ----------------------------------------------------------------------------#
# Main code
# ----------------------------------------------------------------------------#
# --------------- Config -------------------- #
ewc_words = {}  # bag of words from the ewc description
item_words = {}  # bag of words from the item description
all_unique_words = []
word_sim_dict = {}
rows = []
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


# build word similarity and information content cache
print("unique words: {}".format(len(all_unique_words)))


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

sql = """INSERT INTO """ + db_table + """(word1, word2, similarity) VALUES (%s, %s, %s)"""

try:
    cursor.executemany(sql, rows)
    cnx.commit()
except mysql.connector.Error as e:
    print("x Failed inserting data: {}\n".format(e))

endtime = datetime.datetime.now()
print("start  :%s" % starttime)
print("end    :%s" % endtime)
print("elapsed:%s" % (endtime - starttime))
