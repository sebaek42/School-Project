#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 22:19:45 2020

@author: baegseungho
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import pymysql
import time


def load_dbscore_data_one():
    conn = pymysql.connect(host='localhost', user='root', password='password', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from db_score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    curs.close()
    conn.close()
    
    #X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = [ ( t['midterm'] ) for t in data ]
    X = np.array(X)
    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X, y


def gradient_descent_vectorized(X, y):
    epochs = 100000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m = 0.0
    c = 0.0
    
    n = len(y)
    
    c_grad = 0.0
    m_grad = 0.0
    c_list = []
    m_list = []
    for epoch in range(epochs):    
        y_pred = m * X + c
        m_grad = (2*(y_pred - y)*X).sum()/n
        c_grad = (2 * (y_pred - y)).sum()/n
        if epoch % 100 == 0:
            c_list.append(c)
            m_list.append(m)
        
        m = m - learning_rate * m_grad
        c = c - learning_rate * c_grad        
        if ( epoch % 1000 == 0):
            print("epoch %d: m_grad=%f, c_grad=%f, m=%f, c=%f" %(epoch, m_grad, c_grad, m, c) )
    
        if ( abs(m_grad) < min_grad and abs(c_grad) < min_grad ):
            break
    return np.array(m_list), np.array(c_list)
     
def init():
    line.set_data([],[])
    return line,
def animate(i):
    y_pred = m[i] * X + c[i]
    line.set_data(X, y_pred)
    return line,

X, y = load_dbscore_data_one()
m, c = gradient_descent_vectorized(X, y)
fig, ax = plt.subplots()
        
ax.set_xlim(min(X)-2, max(X)+4)
ax.set_ylim(min(y)-2, max(y)+4)
ax.set_xlabel('midterm')
ax.set_ylabel('score')
line, = ax.plot([], [], color='red')
ax.scatter(X, y)
ani = animation.FuncAnimation(fig, animate,init_func=init, frames=len(m), interval=10, blit=True)
plt.show()


def load_dbscore_data_two():
    conn = pymysql.connect(host='localhost', user='root', password='password', db='university')
    curs = conn.cursor(pymysql.cursors.DictCursor)
    
    sql = "select * from db_score"
    curs.execute(sql)
    
    data  = curs.fetchall()
    curs.close()
    conn.close()
    
    X = [ (t['attendance'], t['homework'], t['midterm'] ) for t in data ]
    X = np.array(X)
    x1 = [ (t['attendance']) for t in data ]
    x1 = np.array(x1)
    x2 = [ (t['homework']) for t in data ]
    x2 = np.array(x2)
    x3 = [ (t['midterm']) for t in data ]
    x3 = np.array(x3)

    
    y = [ (t['score']) for t in data]
    y = np.array(y)

    return X,x1,x2,x3, y

def least_square_data(X, y):
    import statsmodels.api as sm
    X_const = sm.add_constant(X) # 모델만들고, least square 적용해야함
    model = sm.OLS(y, X_const) # ordinary lesat square(out,in)
    ls = model.fit()
    print(ls.summary())
    # 각 coefficient나옴
    ls_c = ls.params[0] # c
    ls_m1 = ls.params[1] # m1 attendandce
    ls_m2 = ls.params[2] # m2 homework
    ls_m3 = ls.params[3] # m3 midterm
    return ls_c, ls_m1, ls_m2, ls_m3
    
def m_gradient_descent_vectorized(x1,x2,x3, y):
    epochs = 1800000
    min_grad = 0.0001
    learning_rate = 0.001
    
    m1 = 0.0
    m2 = 0.0
    m3 = 0.0
    c = 0.0
    n = len(y)
    c_grad = 0.0
    m1_grad = 0.0
    m2_grad = 0.0
    m3_grad = 0.0
    for epoch in range(epochs):    
        y_pred = m1*x1 + m2*x2 + m3*x3 + c
        m1_grad = (2*(y_pred - y)*x1).sum()/n
        m2_grad = (2*(y_pred - y)*x2).sum()/n
        m3_grad = (2*(y_pred - y)*x3).sum()/n
        c_grad = (2 * (y_pred - y)).sum()/n
        
        m1 = m1 - learning_rate * m1_grad
        m2 = m2 - learning_rate * m2_grad
        m3 = m3 - learning_rate * m3_grad
        c = c - learning_rate * c_grad        
        if ( epoch % 10000 == 0):
            print("epoch %d: m1_grad=%f, m2_grad=%f, m3_grad=%f ,c_grad=%f, m1=%f, m2=%f, m3=%f,c=%f" %(epoch, m1_grad, m2_grad, m3_grad, c_grad, m1, m2, m3, c) )
    
        if ( abs(m1_grad) < min_grad and abs(m2_grad) < min_grad and abs(m3_grad) < min_grad and abs(c_grad) < min_grad ):
            break
    return m1,m2,m3,c

X, x1,x2,x3, y = load_dbscore_data_two()
ls_c, ls_m1, ls_m2, ls_m3 = least_square_data(X, y)
start_time = time.time()
m1, m2, m3, c = m_gradient_descent_vectorized(x1,x2,x3, y)
end_time = time.time()

print("%f seconds" %(end_time - start_time))

print("\n\nFinal:")
print("gdv_m1=%f, gdv_m2=%f, gdv_m3=%f, gdv_c=%f" %(m1,m2,m3,c))
print("ls_m1=%f, ls_m2=%f, ls_m3=%f, ls_c=%f" %(ls_m1, ls_m2, ls_m3, ls_c))
