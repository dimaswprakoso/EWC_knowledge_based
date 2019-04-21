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
# Similarity Functions
# ----------------------------------------------------------------------------#
def NLP(data):
    # 0. Lowercase
    data = data.lower()

    # 1. Tokenize # word tokenize (removes also punctuation)
    tokenizer = RegexpTokenizer(r'[a-zA-Z_]+')
    words = tokenizer.tokenize(data)

    # 2. Remove short words
    words = [w for w in words if len(w) > 3]

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


def cos_similarity(item_vec1, item_vec2):
    sim = 1 - spatial.distance.cosine(item_vec1, item_vec2)  # cosine similarity

    return sim


# ----------------------------------------------------------------------------#
# Li's sentence similarity
# ----------------------------------------------------------------------------#
ALPHA = 0.2
BETA = 0.45
ETA = 0.4
PHI = 0.2
DELTA = 0.85

brown_freqs = dict()
N = 0

def length_dist(synset_1, synset_2):
    """
    Return a measure of the length of the shortest path in the semantic
    ontology (Wordnet in our case as well as the paper's) between two
    synsets.
    """
    l_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return 0.0
    if synset_1 == synset_2:
        # if synset_1 and synset_2 are the same synset return 0
        l_dist = 0.0
    else:
        wset_1 = set([str(x.name()) for x in synset_1.lemmas()])
        wset_2 = set([str(x.name()) for x in synset_2.lemmas()])
        if len(wset_1.intersection(wset_2)) > 0:
            # if synset_1 != synset_2 but there is word overlap, return 1.0
            l_dist = 1.0
        else:
            # just compute the shortest path between the two
            l_dist = synset_1.shortest_path_distance(synset_2)
            if l_dist is None:
                l_dist = 0.0
    # normalize path length to the range [0,1]
    return math.exp(-ALPHA * l_dist)


def hierarchy_dist(synset_1, synset_2):
    """
    Return a measure of depth in the ontology to model the fact that
    nodes closer to the root are broader and have less semantic similarity
    than nodes further away from the root.
    """
    h_dist = sys.maxsize
    if synset_1 is None or synset_2 is None:
        return h_dist
    if synset_1 == synset_2:
        # return the depth of one of synset_1 or synset_2
        h_dist = max([x[1] for x in synset_1.hypernym_distances()])
    else:
        # find the max depth of least common subsumer
        hypernyms_1 = {x[0]: x[1] for x in synset_1.hypernym_distances()}
        hypernyms_2 = {x[0]: x[1] for x in synset_2.hypernym_distances()}
        lcs_candidates = set(hypernyms_1.keys()).intersection(
            set(hypernyms_2.keys()))
        if len(lcs_candidates) > 0:
            lcs_dists = []
            for lcs_candidate in lcs_candidates:
                lcs_d1 = 0
                if lcs_candidate in hypernyms_1:
                    lcs_d1 = hypernyms_1[lcs_candidate]
                lcs_d2 = 0
                if lcs_candidate in hypernyms_2:
                    lcs_d2 = hypernyms_2[lcs_candidate]
                lcs_dists.append(max([lcs_d1, lcs_d2]))
            h_dist = max(lcs_dists)
        else:
            h_dist = 0
    return ((math.exp(BETA * h_dist) - math.exp(-BETA * h_dist)) /
            (math.exp(BETA * h_dist) + math.exp(-BETA * h_dist)))


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
                if sim == None:
                    sim = 0
                if sim > max_sim:
                    max_sim = sim
                    best_pair = synset_1, synset_2, max_sim
        return best_pair
        # return synsets_1[0], synsets_2[0]


def word_similarity(word_1, word_2):
    synset_pair = get_best_synset_pair(word_1, word_2)
    # return (length_dist(synset_pair[0], synset_pair[1]) *
    #         hierarchy_dist(synset_pair[0], synset_pair[1]))
    return synset_pair[2]


def most_similar_word(word, word_set):
    """
    Find the word in the joint word set that is most similar to the word
    passed in. We use the algorithm above to compute word similarity between
    the word and each word in the joint word set, and return the most similar
    word and the actual similarity value.
    """
    max_sim = -1.0
    sim_word = ""
    for ref_word in word_set:
        sim = word_similarity(word, ref_word)
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
    semvec = np.zeros(len(joint_words))
    i = 0
    for joint_word in joint_words:
        if joint_word in sent_set:
            # if word in union exists in the sentence, s(i) = 1 (unnormalized)
            semvec[i] = 1.0
            # if info_content_norm:
            #     semvec[i] = semvec[i] * math.pow(info_content(joint_word), 2)
        else:
            # find the most similar word in the joint set and set the sim value
            sim_word, max_sim = most_similar_word(joint_word, sent_set)
            semvec[i] = max_sim if max_sim > PHI else 0.0
            # if info_content_norm:
            #     semvec[i] = semvec[i] * info_content(joint_word) * info_content(sim_word)
        i = i + 1

    # return list of item vectors
    vec = semvec.tolist()
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


