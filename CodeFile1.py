# -*- coding: utf-8 -*-
"""
Created on Thu Oct  2 14:03:42 2025

@author: Isaac Bradford, ibradfo@purdue.edu
Code created for instruction on pulling data and making a graph
"""

#import our packages
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy as sp

#https://stackoverflow.com/questions/25579227/seaborn-lmplot-with-equation-and-r2-text
#r2 addtion to graphs

# df = dataframe, pd.read_csv() reads csv file using pandas
df = pd.read_csv('DataForCode1.csv')

print(df) # printing to visibility

# "NOx vs PO4" # Scatterplot
#make a scatterplot with correct labels
plt.figure(figsize=(12,6)) #space out recording #matplotlib
graphy = sns.scatterplot(x='NOx', y='PO4', data=df) #uses seaborn to graph
graphy.set_xlabel('NOx (ppm)') # change text of x label
graphy.set_ylabel('PO4 (ppm)') #change text of y label
graphy.set_title('NOx vs PO4 Concentration in 2024') # set title to text option
plt.savefig('Graphy.png', dpi = 300) #save figure to folder of repository
plt.show() # show graph in viewer

#lmplot of NOx nd PO4
g = sns.lmplot(x='NOx', y='PO4', data=df)
#plt.set_title('NOx vs. PO4')

def annotate(data, **kws):
    r, p = sp.stats.pearsonr(df['NOx'], df['PO4']) #creating r and p from pearson stats 
    ax = plt.gca()  #get current axes
    ax.text(.05, .8, 'r={:.2f}, p={:.2g}'.format(r, p), transform=ax.transAxes) #inserting r and p on graph
    rs = r ** 2
    ax.text(.05, .7, 'r2={rs}'.format(rs), transform=ax)

"""
x = df['NOx']
y = df['PO4']

def r2(x, y):
    return sp.stats.pearsonr(x, y)[0] ** 2
sns.jointplot(x, y, kind="reg", stat_func=r2)
"""
g.map_dataframe(annotate)
plt.show()

