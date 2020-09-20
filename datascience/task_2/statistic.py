def print_mean(df):
    col = df.columns
    i = 0
    print("[ Mean value ]")
    while i < len(col) - 1:
        print(str(col[i]) + " : " + str(df[col[i]].mean()))
        i = i + 1
    print()

def print_median(df):
    col = df.columns
    i = 0
    print("[ Median value ]")
    while i < len(col) - 1:
        print(str(col[i]) + " : " + str(df[col[i]].median()))
        i = i + 1
    print()

def print_mode(df):
    col = df.columns
    i = 0
    print("[ Mode value ]")
    result = df[col[7]].mode()
    while i < len(result):
        print(result[i], end = ' ')
        i = i + 1
    print()
    print()

def print_spread(df):
    col = df.columns
    i = 0
    print("[ Variance value ]")
    while i < len(col) - 1:
        print(str(col[i]) + " : " + str(df[col[i]].var()))
        i = i + 1
    print()
    i = 0
    print("[ standard deviation value ]")
    while i < len(col) - 1:
        print(str(col[i]) + " : " + str(df[col[i]].std()))
        i = i + 1
    print()
    i = 0
    print("[ AAD value ]")
    while i < len(col) - 1:
        temp = df[col[i]]
        temp_mean = df[col[i]].mean()
        aad = abs(temp - temp_mean).mean()
        print(str(col[i]) + " : " + str(aad))
        i = i + 1
    print()
    i = 0
    print("[ MAD value ]")
    while i < len(col) - 1:
        temp = df[col[i]]
        temp_mean = df[col[i]].mean()
        mad = abs(temp - temp_mean).median()
        print(str(col[i]) + " : " + str(mad))
        i = i + 1
    print()

def print_pplot(df):
    col = df.columns
    p = np.linspace(0, 100)
    print("[ Percentile plot ]")
    i = 0
    while i < len(col) - 1:
        y = df[col[i]]
        ax = plt.gca()
        ax.plot(p, np.percentile(y, p, interpolation='linear'))
        ax.set(title=col[i] + ' percentile plot',
              xlabel='Percentile',
              ylabel=col[i])
        plt.show()
        i = i + 1
    print()

def print_boxplot(df):
    col = df.columns
    i = 0
    print("[ Box plot ]")
    while i < len(col) - 1:
        target = df[col[i]]
        plt.boxplot(target)
        plt.title(col[i])
        plt.show()
        i = i + 1

def print_histogram(df):
    col = df.columns
    i = 0
    print("[ Histogram ]")
    while i < len(col):
        target = df[col[i]]
        plt.hist(target)
        plt.title(col[i], pad=10)
        plt.ylabel('frequency', labelpad=10)
        plt.show()
        i = i + 1

def print_splot(df):
    copy = df
    del copy['grade']
    col = copy.columns
    combination = list(combinations(col, 2))
    i = 0
    print("[ Scatter plot ]")
    while i < len(combination):
        target = combination[i]
        X = target[0]
        Y = target[1]
        plt.scatter(df[X], df[Y])
        plt.title(X + ' & ' + Y, pad=10)
        plt.xlabel(X, labelpad=10)
        plt.ylabel(Y, labelpad=10)
        plt.show()
        i = i + 1

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations

x_file = '~/db_score.xlsx'
df = pd.read_excel(x_file)
print_mean(df)
print_median(df)
print_mode(df)
print_spread(df)

# percentile plot(grade x)
print_pplot(df)

# box plot (grade x)
print_boxplot(df)

# histogram (all)
print_histogram(df)

# scatter plot (every possible combination)
print_splot(df)
