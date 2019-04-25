import mysql.connector
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from scipy import spatial
from collections import OrderedDict
from operator import itemgetter
import math
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
# Get word similarity cache
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `word_sim_cache`
"""

word_sim_cache = {}

try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        word_sim_cache[row['word1'], row['word2']] = row['similarity']

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

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

    # lm = WordNetLemmatizer()
    # lemmatized_words = []
    # for w in words:
    #     lemmatized_words.append(lm.lemmatize(w))
    # data = lemmatized_words

    return data


def find_unique_words(data, data2):
    # merge words from all items, then remove duplicates with set datatype
    all_unique_words = []
    all_unique_words = list(set(all_unique_words + data))

    for i, j in data2.items():
        all_unique_words = list(set(all_unique_words + j))

    return all_unique_words


def cos_similarity(item_vec1, item_vec2):
    sim = 1 - spatial.distance.cosine(item_vec1, item_vec2)  # cosine similarity

    return sim


def most_similar_word(word, word_set):
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = word_sim_cache[word, ref_word]
        if sim > max_sim:
            max_sim = sim
            sim_word = ref_word
    return sim_word, max_sim


def gen_item_vector(data, all_unique_words):
    # create a list of item vectors, initialize each item vector with zero values
    vec = {}
    vec = [0] * len(all_unique_words)

    sent_set = set(data)
    joint_words = set(all_unique_words)
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            vec[i] = 1.0
            # if info_content_norm:
            #     vec[i] = vec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            vec[i] = max_sim if max_sim > PHI else 0.0
            # if info_content_norm:
            #     vec[i] = vec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1

    # return list of item vectors
    return vec



# ----------------------------------------------------------------------------#
# 3. Recommendation
# ----------------------------------------------------------------------------#
def recommend(item_desc, ewc_words):

    item_vec1 = {}
    item_vec2 = {}
    sim_list = {}

    # uw = find_unique_words(item_desc, ewc_words)
    it = 0

    # match the words from the item description against the words of each EWC code description
    for k, l in ewc_words.items():
        # Lets do some matching -->
        # build vector dimension by joining item description and current ewc only, not the whole ewc catalog
        uw = find_unique_words(item_desc, {k:l})
        item_vec1[it] = gen_item_vector(l, uw)  # ewc code vector
        item_vec2[it] = gen_item_vector(item_desc, uw)  # item desc vector

        # check if item vector is not empty
        if sum(item_vec1[it]) > 0 and sum(item_vec2[it]) > 0:
            sim_list[k] = cos_similarity(item_vec1[it], item_vec2[it])

        it += 1

    return sim_list


def generate_recommendation_list(sim_matrix, top_n, min_sim):
    # default value
    if top_n is None:
        top_n = 1

    rec_list = {}
    top_itter = 0

    s = [(k, sim_matrix[k]) for k in sorted(sim_matrix, key=sim_matrix.get, reverse=True)]
    for k, v in s:
        # top 10 and similarity is high enough
        if top_itter < top_n and v > min_sim:
            rec_list[k] = v
        top_itter += 1

    rec_list = OrderedDict(sorted(rec_list.items(), key=itemgetter(1), reverse=True))

    return rec_list


# ----------------------------------------------------------------------------#
# 4. Evaluation
# ----------------------------------------------------------------------------#
def eval_topn(rec_list, ewc):
    it = 1
    m = {}
    m['no_rec'] = len(rec_list)  # how many recommendations were provided
    m['correct'] = 0  # was the right recommendation in the list
    m['position'] = 0  # What was the position (no 2 out of 10) of the right
    m['ewc_label'] = 1  # Some items have '99 99 99', thus no EWC code assigned. Needed for EWC

    for i, j in rec_list.items():
        # print(i+" -<>- "+ewc)

        if i == ewc:
            m['correct'] = 1
            m['position'] = it
        if ewc == '99 99 99':
            m['ewc_label'] = 0

        it += 1

    return m


def eval_recommendations(ev):
    m = {}  # dictionary having all performance metrics

    m['no_items'] = len(ev)  # number of items for which recommendation could be provided

    m['no_rec'] = 0  # total recommendations provided
    for i, j in ev.items():
        if j['no_rec'] > 0:
            m['no_rec'] += 1

    m['no_labeled'] = 0
    for i, j in ev.items():
        if j['ewc_label'] == 1:
            m['no_labeled'] += 1

    m['tp'] = 0  # True positives (inherent to all correct recommendations)
    for i, j in ev.items():
        if j['correct'] > 0:
            m['tp'] += 1

    m['fp'] = 0  # False positives (inherent to incorrect recommendations)
    for i, j in ev.items():
        if j['correct'] == 0:
            m['fp'] += 1

    m['tn'] = 0  # True negatives ()
    for i, j in ev.items():
        if j['ewc_label'] == 1 and j['no_rec'] == 0:
            m['tn'] += 1

    m['fn'] = 0  # False negatives ()
    for i, j in ev.items():
        if j['correct'] == 0 and j['no_rec'] == 0:
            m['fn'] += 1

    # --------------------------------- #

    # Precision = TP / TP + FP
    m['precision'] = m['tp'] / (m['tp'] + m['fp'])

    # Recall = TP / no_items_with_ewc_label
    m['recall'] = m['tp'] / m['no_labeled']

    # Accuracy = TP + True Negatives / all items
    m['accuracy'] = (m['tp'] + m['tn']) / m['no_items']

    # F1 measure
    if m['precision'] + m['recall'] > 0:
        m['f1'] = 2 * ((m['precision'] * m['recall']) / (m['precision'] + m['recall']))
    else:
        m['f1'] = 0

    # Return all measures
    return m


# ----------------------------------------------------------------------------#
# 5. Main code
# ----------------------------------------------------------------------------#
# --------------- Config -------------------- #
top_n = 5  # Maximal number of recommendations in the recommendation set.
min_sim = 0.1  # higher than zero

ev = {}  # dictionary containing all evaluations of recommendations
ewc_words = {}  # bag of words from the ewc description
item_words = {}  # bag of words from the item description
rec = {}  # dictionary containing the recommendations for an item desc
sim_matrix = {}  # the similarity matrix between the vectors of ewc desc and item desc
PHI = 0.8 #word similarity threshold
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

# prepare the ewc_desc
for k, l in ewc_list.items():
    ewc_words[k] = NLP(l[0])

# for all items
for m, n in item_list.items():
    # Generate the similarity matrix
    sim_matrix = recommend(item_words[m], ewc_words)

    # generate the list of recommendations for an item description
    rec = generate_recommendation_list(sim_matrix, top_n, min_sim)

    # Evaluate if the recommendation was correct (with stats)
    ev[m] = eval_topn(rec, item_list[m][1])

# Calculate the performance metrics (e.g. accuracy, precision, recall, F1) over all items
print(ev)
results = eval_recommendations(ev)
print(results)

end = time.time()
print(end - start)


