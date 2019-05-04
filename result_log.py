import mysql.connector

# rst = {'test_type':'wup-lemma', 'pos':'n', 'wordsim_th':0.8, 'top_n':10, 'duration':808.5, 'no_items':356,
#        'no_rec':346, 'no_labeled':312, 'tp':74, 'fp':282, 'tn':10, 'fn':10, 'precision':0.20786516853932585,
#        'recall':0.23717948717948717, 'accuracy':0.23595505617977527, 'f1':0.2215568862275449}



def log_result(log_data):
    db_user = 'root'
    db_database = 'sharebox'
    language = 'EN'
    cnx = mysql.connector.connect(user=db_user, database=db_database)
    cursor = cnx.cursor(dictionary=True)

    columns = ','.join("`" + str(x).replace('/', '_') + "`" for x in log_data.keys())
    values = ','.join("'" + str(x).replace('/', '_') + "'" for x in log_data.values())
    sql = "INSERT INTO %s (test_time, %s) VALUES (NOW(), %s );" % ('result_log', columns, values)

    try:

        cursor.execute(sql)
        cnx.commit()
    except mysql.connector.Error as e:
        print("x Failed inserting data: {}\n".format(e))


# log_result(rst)
# print("done")
