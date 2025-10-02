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


