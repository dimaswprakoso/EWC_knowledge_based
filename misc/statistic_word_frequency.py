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
import nltk
import numpy as np
import matplotlib.pyplot as plt

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
    # term_list = ['waste', 'wastes', 'scrap', 'scraps', 'process', 'consultancy', 'advice', 'training', 'service', 'managing',
    #              'management', 'recycling', 'recycle', 'industry', 'industrial', 'material', 'quantity', 'support',
    #              'residue', 'organic', 'remainder']
    # words = [w for w in words if not w in term_list]  # for each word check if

    # 5. Find Stem # Porter Stemmer
    # ps = PorterStemmer()
    # stemmed_words = []
    # for w in words:
    #     stemmed_words.append(ps.stem(w))
    # data = stemmed_words

    # lm = WordNetLemmatizer()
    # lemmatized_words = []
    # for w in words:
    #     lemmatized_words.append(lm.lemmatize(w))
    # data = lemmatized_words
    data = words

    return data


# ----------------------------------------------------------------------------#
# 1. Get items from workshop
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `workshop_items2` WHERE language = '""" + language + """' AND type='Material' 
"""

# RTRIM(LTRIM(wastecode)) != '99 99 99' LIMIT 0,25

# item_list = {}
item_list = []
try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        # item_list[row['id']] = [row['Waste_description'], row['Wastecode']]
        item_list = item_list + NLP(row['Waste_description'].strip())

    print("Items rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

# ----------------------------------------------------------------------------#
# 2. Get items from ewc
# ----------------------------------------------------------------------------#
sql = """
SELECT * FROM `ewc_level3`
"""

ewc_list = []

try:
    results = cursor.execute(sql)
    rows = cursor.fetchall()

    for row in rows:
        ewc_list = ewc_list + NLP(row['description'].strip())

    print("EWC rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))

all_words = item_list + ewc_list

fdist = nltk.FreqDist(all_words)


for word, frequency in fdist.most_common(20):
    print(u'{}:{}'.format(word, frequency))
    print(frequency)

print(fdist)
fdist.plot(40, cumulative=True)
plt.show()
#

