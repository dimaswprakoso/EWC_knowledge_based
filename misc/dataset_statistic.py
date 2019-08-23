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
# import matplotlib.mlab as mlab


# ----------------------------------------------------------------------------#
# Configuration
# ----------------------------------------------------------------------------#
start = time.time()

db_user = 'root'
db_database = 'sharebox'
language = 'EN'
lens = []

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
        words = NLP(row['Waste_description'].strip())
        lens.append(len(words))
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
        words = NLP(row['description'].strip())
        lens.append(len(words))
        ewc_list = ewc_list + words

    print("EWC rows: {}".format(cursor.rowcount))

except mysql.connector.Error as e:
    print("x Failed loading data: {}\n".format(e))


# Plot.
plt.rc('figure', figsize=(8,6))
plt.rc('font', size=20)
plt.rc('lines', linewidth=1)
# plt.rc('axes', prop_cycle=('#377eb8','#e41a1c','#4daf4a','#984ea3','#ff7f00','#ffff33'))

# Histogram.
plt.hist(lens, bins='auto', rwidth = 5)
# plt.hold(True)
# Average length.
avg_len = sum(lens) / float(len(lens))
plt.axvline(avg_len, color='#e41a1c')
# plt.hold(False)
plt.title('Histogram of sentence lengths.')
plt.xlabel('Length')
plt.ylabel('Frequency')
plt.text(5, 160, 'mean = %.2f' % avg_len)
plt.show()



