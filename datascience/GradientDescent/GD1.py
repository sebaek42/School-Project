#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 16:07:11 2020

@author: baegseungho
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pymysql



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
