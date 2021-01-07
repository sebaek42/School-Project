#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 18:56:32 2020

@author: baegseungho
"""

import pymysql
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing


def load_data():
    file = 'db_score_3_labels.xlsx' #####경로확인
    db_score = pd.read_excel(file)
    
    conn = pymysql.connect(host='localhost', user='root', password='password', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    drop_sql = """drop table if exists db_score_3_labels""" ####테이블확인
    curs.execute(drop_sql)
    conn.commit()
    
    import sqlalchemy
    database_username = 'root'
    database_password = 'password'
    database_ip = 'localhost'
    database_name = 'university'
    database_connection = sqlalchemy.create_engine('mysql+pymysql://{0}:{1}@{2}/{3}'.format(database_username, database_password, database_ip, database_name))
    db_score.to_sql(con=database_connection, name='db_score_3_labels', if_exists='replace') ###name 확인


def performance_eval(y, y_predict, flag):
    if flag == 0:
        tp, tn, fp, fn = 0,0,0,0
        for y, yp in zip(y, y_predict):
            if y == 1 and yp == 1:
                tp += 1
            elif y == 1 and yp == -1:
                fn += 1 
            elif y == -1 and yp == 1:
                fp += 1
            else:
                tn += 1
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        if tp + fp != 0:
            precision = tp/(tp+fp)
        else:
            precision = 0
        if tp+fn != 0:
            recall = tp/(tp+fn)
        else:
            recall = 0
        if precision + recall != 0:
            f1_score = 2*precision*recall/(precision+recall)
        else:
            f1_score = 0
        return accuracy, precision, recall, f1_score
    else:
        i = 0
        tp = [0,0,0]
        tn = [0,0,0]
        fp = [0,0,0]
        fn = [0,0,0]
        y_temp = y
        yp_temp = y_predict
        while i < 3:
            y = y_temp
            y_predict = yp_temp
            for y, yp in zip(y, y_predict):
                if y == i and yp == i:
                    tp[i] += 1
                elif y == i and yp != i:
                    fn[i] += 1 
                elif y != i and yp == i:
                    fp[i] += 1
                elif y != i and yp != i:
                    tn[i] += 1
            i += 1
        i = 0
        num = 0
        dnum = 0
        while i < 3:
            num += tp[i]
            i += 1
        accuracy = num/len(y_predict)
        
        i = 0
        num = 0
        dnum = 0
        precision = [0,0,0]
        recall = [0,0,0]
        f1_score = [0,0,0]
        while i < 3:
            num += tp[i]
            dnum += tp[i]+fp[i]
            precision[i] = num/dnum
            i += 1
    
        i = 0
        num = 0
        dnum = 0
        while i < 3:
            num += tp[i]
            dnum += tp[i]+fn[i]
            if dnum != 0:
                recall[i] = num/dnum
            else:
                recall[i] = 0
            i += 1
        
        i = 0
        while i < 3:
            if precision[i] + recall[i] != 0:
                f1_score[i] = 2*precision[i]*recall[i]/(precision[i]+recall[i])
            else:
                f1_score[i] = 0
            i += 1
    
        return accuracy, precision, recall, f1_score  

def train_split(X, y, flag):
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.33, random_state=42)
    print("SVM")
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_predict = clf.predict(X_test)
    accuracy, precision, recall, f1_score = performance_eval(y_test, y_predict, flag)
    print("accuracy=%f" %accuracy)
    if flag == 0:
        print("precision=%f" %precision)
        print("recall=%f" %recall)
        print("f1_score=%f" %f1_score)
        print()
    else:
        print("Grade A")
        print("precision=%f" %precision[0])
        print("recall=%f" %recall[0])
        print("f1_score=%f" %f1_score[0])
        print()
        print("Grade B")
        print("precision=%f" %precision[1])
        print("recall=%f" %recall[1])
        print("f1_score=%f" %f1_score[1])
        print()
        print("Grade C")
        print("precision=%f" %precision[2])
        print("recall=%f" %recall[2])
        print("f1_score=%f" %f1_score[2])
        print()
    
    print("Logistic Regression")
    model = LogisticRegression(max_iter=50000, C=10, random_state=42)
    model.fit(X_train, y_train)
    y_predict = model.predict(X_test)
    accuracy, precision, recall, f1_score = performance_eval(y_test, y_predict, flag)
    print("accuracy=%f" %accuracy)
    if flag == 0:
        print("precision=%f" %precision)
        print("recall=%f" %recall)
        print("f1_score=%f" %f1_score)
        print()
    else:
        print("Grade A")
        print("precision=%f" %precision[0])
        print("recall=%f" %recall[0])
        print("f1_score=%f" %f1_score[0])
        print()
        print("Grade B")
        print("precision=%f" %precision[1])
        print("recall=%f" %recall[1])
        print("f1_score=%f" %f1_score[1])
        print()
        print("Grade C")
        print("precision=%f" %precision[2])
        print("recall=%f" %recall[2])
        print("f1_score=%f" %f1_score[2])
        print() 
    print("Random Forest")
    forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
    forest.fit(X_train, y_train)
    y_predict = forest.predict(X_test)
    accuracy, precision, recall, f1_score = performance_eval(y_test, y_predict, flag)
    print("accuracy=%f" %accuracy)
    if flag == 0:
        print("precision=%f" %precision)
        print("recall=%f" %recall)
        print("f1_score=%f" %f1_score)
        print()
    else:
        print("Grade A")
        print("precision=%f" %precision[0])
        print("recall=%f" %recall[0])
        print("f1_score=%f" %f1_score[0])
        print()
        print("Grade B")
        print("precision=%f" %precision[1])
        print("recall=%f" %recall[1])
        print("f1_score=%f" %f1_score[1])
        print()
        print("Grade C")
        print("precision=%f" %precision[2])
        print("recall=%f" %recall[2])
        print("f1_score=%f" %f1_score[2])
        print()

def k_fold_cross_validation(X, y, flag):
    from sklearn.model_selection import KFold
    kf = KFold (n_splits=5, random_state=42, shuffle=True)
    s_accuracy = []
    l_accuracy = []
    f_accuracy = []
    if flag == 1:
        s_precision = [[],[],[]]
        s_recall = [[],[],[]]
        s_f1_score = [[],[],[]]
        l_precision = [[],[],[]]
        l_recall = [[],[],[]]
        l_f1_score = [[],[],[]]
        f_precision = [[],[],[]]
        f_recall = [[],[],[]]
        f_f1_score = [[],[],[]]
    else:
        s_precision = []
        s_recall = []
        s_f1_score = []
        l_precision = []
        l_recall = []
        l_f1_score = []
        f_precision = []
        f_recall = []
        f_f1_score = []

    for train_index, test_index in kf.split(X): 
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        i = 0
        while i < 3:
            if i == 0:
                clf = svm.SVC(kernel='linear')
                clf.fit(X_train, y_train)
                y_predict = clf.predict(X_test)
                acc, prec, rec, f1 = performance_eval(y_test, y_predict, flag)
                s_accuracy.append(acc)
                if flag == 0:
                    s_precision.append(prec)
                    s_recall.append(rec)
                    s_f1_score.append(f1)
                else:
                    j = 0
                    while j < 3:
                        s_precision[j].append(prec[j])
                        s_recall[j].append(rec[j])
                        s_f1_score[j].append(f1[j])
                        j += 1
            elif i == 1:
                model = LogisticRegression(max_iter=50000, C=10, random_state=42)
                model.fit(X_train, y_train)
                y_predict = model.predict(X_test)
                acc, prec, rec, f1 = performance_eval(y_test, y_predict, flag)
                l_accuracy.append(acc)
                if flag == 0:
                    l_precision.append(prec)
                    l_recall.append(rec)
                    l_f1_score.append(f1)
                else:
                    j = 0
                    while j < 3:
                        l_precision[j].append(prec[j])
                        l_recall[j].append(rec[j])
                        l_f1_score[j].append(f1[j])
                        j += 1
            else:
                forest = RandomForestClassifier(n_estimators=100, oob_score=True, random_state=42)
                forest.fit(X_train, y_train)
                y_predict = forest.predict(X_test)
                acc, prec, rec, f1 = performance_eval(y_test, y_predict, flag)
                f_accuracy.append(acc)
                if flag == 0:
                    f_precision.append(prec)
                    f_recall.append(rec)
                    f_f1_score.append(f1)
                else:
                    j = 0
                    while j < 3:
                        f_precision[j].append(prec[j])
                        f_recall[j].append(rec[j])
                        f_f1_score[j].append(f1[j])
                        j += 1
            i += 1
    
    import statistics
    print("SVM")
    print("average_accuracy =", statistics.mean(s_accuracy))
    if flag == 0:
        print("average_precision =", statistics.mean(s_precision))
        print("average_recall =", statistics.mean(s_recall))
        print("average_f1_score =", statistics.mean(s_f1_score))
        print()
    else:
        j = 0
        while j < 3:
            if j == 0:
                print("Grade A")
                print("average_precision =", statistics.mean(s_precision[j]))
                print("average_recall =", statistics.mean(s_recall[j]))
                print("average_f1_score =", statistics.mean(s_f1_score[j]))
                print()
            elif j == 1:
                print("Grade B")
                print("average_precision =", statistics.mean(s_precision[j]))
                print("average_recall =", statistics.mean(s_recall[j]))
                print("average_f1_score =", statistics.mean(s_f1_score[j]))
                print()
            else:
                print("Grade C")
                print("average_precision =", statistics.mean(s_precision[j]))
                print("average_recall =", statistics.mean(s_recall[j]))
                print("average_f1_score =", statistics.mean(s_f1_score[j]))
                print()
            j += 1
    
    print("Logistic Regression")
    print("average_accuracy =", statistics.mean(l_accuracy))
    if flag == 0:
        print("average_precision =", statistics.mean(l_precision))
        print("average_recall =", statistics.mean(l_recall))
        print("average_f1_score =", statistics.mean(l_f1_score))
        print()
    else:
        j = 0
        while j < 3:
            if j == 0:
                print("Grade A")
                print("average_precision =", statistics.mean(l_precision[j]))
                print("average_recall =", statistics.mean(l_recall[j]))
                print("average_f1_score =", statistics.mean(l_f1_score[j]))
                print()
            elif j == 1:
                print("Grade B")
                print("average_precision =", statistics.mean(l_precision[j]))
                print("average_recall =", statistics.mean(l_recall[j]))
                print("average_f1_score =", statistics.mean(l_f1_score[j]))
                print()
            else:
                print("Grade C")
                print("average_precision =", statistics.mean(l_precision[j]))
                print("average_recall =", statistics.mean(l_recall[j]))
                print("average_f1_score =", statistics.mean(l_f1_score[j]))
                print()
            j += 1
    print("Random Forest")
    print("average_accuracy =", statistics.mean(f_accuracy))
    if flag == 0:
        print("average_precision =", statistics.mean(f_precision))
        print("average_recall =", statistics.mean(f_recall))
        print("average_f1_score =", statistics.mean(f_f1_score))
        print()
    else:
        j = 0
        while j < 3:
            if j == 0:
                print("Grade A")
                print("average_precision =", statistics.mean(f_precision[j]))
                print("average_recall =", statistics.mean(f_recall[j]))
                print("average_f1_score =", statistics.mean(f_f1_score[j]))
                print()
            elif j == 1:
                print("Grade B")
                print("average_precision =", statistics.mean(f_precision[j]))
                print("average_recall =", statistics.mean(f_recall[j]))
                print("average_f1_score =", statistics.mean(f_f1_score[j]))
                print()
            else:
                print("Grade C")
                print("average_precision =", statistics.mean(f_precision[j]))
                print("average_recall =", statistics.mean(f_recall[j]))
                print("average_f1_score =", statistics.mean(f_f1_score[j]))
                print()
            j += 1   

def binary_classification(data):
    X = [ (t['homework'], t['discussion'], t['midterm'])  for t in data ]
    X = np.array(X)
    X = preprocessing.scale(X)

    y = [ 1 if (t['grade'] == 'B') else -1 for t in data ]
    y = np.array(y)
    print("======Binary Classification=====")
    print()
    print("-----Train_test_split-----")
    print()
    train_split(X, y, 0)
    print("-----K-Fold-----")
    print()
    k_fold_cross_validation(X, y, 0)


def multi_classification(data):
    X = [ (t['homework'], t['discussion'], t['midterm'])  for t in data ]
    X = np.array(X)
    X = preprocessing.scale(X)
    
    y = [ 0 if (t['grade'] == 'A') else 1 if (t['grade'] == 'B') else 2 for t in data ]
    y = np.array(y)
    print("======Multi Classification=====")
    print()
    print("-----Train_test_split-----")
    print()
    train_split(X, y, 1)
    print("-----K-Fold-----")
    print()
    k_fold_cross_validation(X, y, 1)


load_data()
conn = pymysql.connect(host='localhost', user='root', password='password', db='university')
curs = conn.cursor(pymysql.cursors.DictCursor)
sql = "select homework, discussion, midterm, grade from db_score_3_labels"
curs.execute(sql)
data = curs.fetchall()
curs.close()
conn.close()

binary_classification(data)
multi_classification(data)
