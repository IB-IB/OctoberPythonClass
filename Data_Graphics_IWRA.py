# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 13:04:31 2025
COntents: create graphics for IWRA2025 Conference from NASA-RSWQ Dataset
@author: Isaac Bradford, ibradfo@purdue.edu
"""

# import required modules
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.dates as mdates
import numpy as np
import scipy as sp
import sklearn
from sklearn import linear_model

from sklearn.datasets import fetch_california_housing #imp the fetch
from sklearn.preprocessing import StandardScaler


df = pd.read_csv('Data_2024_All_ForCode.csv') #open a csv within GH folder for data process.

print(df)

df_ch = df.dropna(subset='Chla (ug/L)') #dropna gets rid of nan vlues

'''CHLA Scatterplot'''
vd = df_ch[['Chla (ug/L)', 'Distance from Dam']].groupby('Distance from Dam').mean()
# ^ special function that subets and creates a new 

#print(vd) #checker
plt.figure(figsize=(12,6)) #space out recording
adgb = sns.scatterplot(x='Distance from Dam', y='Chla (ug/L)', data=vd) 
adgb.set_xlabel('Distance from Dam (m)')
adgb.set_ylabel('Average Chl-a Concentration (ug/L)')
adgb.set_title("Average Chl-a Recorded Distance from the Dam over Summer 2024")
plt.show()

'''CHLA Boxes SALA-MISS'''
#https://saturncloud.io/blog/how-to-remove-rows-with-specific-values-in-pandas-dataframe/
#isolate data by Lake
mask_miss = df['Lake ID (4 dig.)'] == 'MISS'
mask_sala = df['Lake ID (4 dig.)'] == 'SALA'
mask_shaf = df['Lake ID (4 dig.)'] == 'SHAF'

df_sh = df_ch[~mask_miss & ~mask_sala]
df_sa = df_ch[~mask_miss & ~mask_shaf]
df_mi = df_ch[~mask_shaf & ~mask_sala]
print('SHAF')
print(df_sh)
print('SALA')
print(df_sa)
print('MISS')
print(df_mi)

plt.figure(figsize=(12,6)) #space out recording
adgb = sns.boxplot(x='Distance from Dam', y='Chla (ug/L)', data=df_sh, showmeans=True, meanprops={'marker':'o','markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':'5'}) 
adgb.set_xlabel('Distance from Dam (m)')
adgb.set_ylabel('Chl-a Concentration (ug/L)')
adgb.set_title("Shafer: Chl-a Recorded Distance from the Dam over Summer 2024")
adgb.set_ylim(0,90)
plt.savefig('ChlaSHAF.png', dpi = 400)
plt.show()

plt.figure(figsize=(12,6)) #space out recording
adgb = sns.boxplot(x='Distance from Dam', y='Chla (ug/L)', data=df_sa, showmeans=True, meanprops={'marker':'o','markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':'5'}) 
adgb.set_xlabel('Distance from Dam (m)')
adgb.set_ylabel('Chl-a Concentration (ug/L)')
adgb.set_title("Salamonie: Chl-a Recorded Distance from the Dam over Summer 2024")
adgb.set_ylim(0,90)
plt.savefig('ChlaSALA.png', dpi = 400)
plt.show()

plt.figure(figsize=(12,6)) #space out recording
adgb = sns.boxplot(x='Distance from Dam', y='Chla (ug/L)', data=df_mi, showmeans=True, meanprops={'marker':'o','markerfacecolor':'black', 'markeredgecolor':'black', 'markersize':'5'}) 
#showmeans adds mean dots
#meanprops sets properties of the mean dots
adgb.set_xlabel('Distance from Dam (m)')
adgb.set_ylabel('Chl-a Concentration (ug/L)')
adgb.set_title("Mississinewa: Chl-a Recorded Distance from the Dam over Summer 2024")
adgb.set_ylim(0,90)
plt.savefig('ChlaMISS.png', dpi = 400)
plt.show()

df_tur = df_ch.dropna(subset='TURBIDITY FNU') #dropna gets rid of nan vlues

gz = sns.jointplot(x='TURBIDITY FNU', y='Chla (ug/L)', data=df_tur, kind='reg')
gz.set_axis_labels(xlabel='Turbidity (FNU)', ylabel='Chla (ug/L)')
plt.suptitle("Turbidity vs. Chla, 2024 All 3 Lakes")
r, p = sp.stats.pearsonr(df_tur['TURBIDITY FNU'], df_tur['Chla (ug/L)'])
ax = plt.gca()
ax.text(.05, .8, 'r={:.2f}'.format(r), transform=ax.transAxes)
#plt.savefig(outFileName) # saves the graph into the github folder
plt.savefig('TurvChla.png', dpi = 400)
plt.show()


mid = pd.read_csv('Data_2024_All_ForCode_MISS.csv') #open a csv within GH folder for data process.


#CHLA
'''-MISS--'''
#dm = mid.dropna(subset='Chla (ug/L)')
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
m_corr = mid.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(m_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Mississinewa Chla Correlation') 
plt.savefig('ChlaHMC_M.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()

mid_r = pd.read_csv('Data_2024_All_ForCode_MISS_RS.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
mr_corr = mid_r.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(mr_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Mississinewa Chla Correlation (RS)') 
plt.savefig('ChlaHMCRS_M.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()


# TDS, ODO in mg/L. Turb in FNU, Temp in C
mid_p = pd.read_csv('Data_2024_All_ForCode_MISS_Phys.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(11, 11))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
mp_corr = mid_p.corr()
mask = np.triu(mp_corr)
np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(mp_corr, annot=True, fmt=".2f", cmap='coolwarm', mask=mask)

plt.title('Correlation Matrix [Physically Collected]') 
plt.savefig('ChlaHMCP_M.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()

sad = pd.read_csv('Data_2024_All_ForCode_SALA.csv') #open a csv within GH folder for data process.

#CHLA
'''-SALA--'''
#dm = mid.dropna(subset='Chla (ug/L)')
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
sa_corr = sad.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(sa_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Salamonie Chla Correlation') 
plt.savefig('ChlaHMC_SA.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()

sad_r = pd.read_csv('Data_2024_All_ForCode_SALA_RS.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
sar_corr = sad_r.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(sar_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Salamonie Chla Correlation (RS)') 
plt.savefig('ChlaHMCRS_SA.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()


# TDS, ODO in mg/L. Turb in FNU, Temp in C
sad_p = pd.read_csv('Data_2024_All_ForCode_SALA_Phys.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(11, 11))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
map_corr = sad_p.corr()
mask = np.triu(map_corr)
np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(map_corr, annot=True, fmt=".2f",cmap='coolwarm', mask=mask)

plt.title('Correlation Matrix [Physically Collected]') 
plt.savefig('ChlaHMCP_SA.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()

shd = pd.read_csv('Data_2024_All_ForCode_SHAF.csv') #open a csv within GH folder for data process.
#CHLA
'''-SHAF--'''
#dm = mid.dropna(subset='Chla (ug/L)')
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
sh_corr = shd.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(sh_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Shafer Chla Correlation') 
plt.savefig('ChlaHMC_SH.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()

shd_r = pd.read_csv('Data_2024_All_ForCode_SHAF_RS.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(4, 8.5))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
sr_corr = shd_r.corr()
#mask = np.triu(m_corr)
#np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(sr_corr[['Chla (ug/L)']], annot=True, cmap='coolwarm')
plt.title('Shafer Chla Correlation (RS)') 
plt.savefig('ChlaHMCRS_SH.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()


# TDS, ODO in mg/L. Turb in FNU, Temp in C
shd_p = pd.read_csv('Data_2024_All_ForCode_SHAF_Phys.csv') #open a csv within GH folder for data process.
#assuming '' is the DataFrame containing the data
#Plotting the correlation matrix
plt.figure(figsize=(11, 11))
#mask half the matrix: https://stackoverflow.com/questions/57414771/how-to-plot-only-the-lower-triangle-of-a-seaborn-heatmap
# TODO Fix Corr Heatmap df_corrm = df_m['NDTIG1', ]
print('WHAT')
sp_corr = shd_p.corr()
mask = np.triu(sp_corr)
np.fill_diagonal(mask, False) #keep the 1=corr diagonal line
sns.heatmap(sp_corr, annot=True, fmt=".2f",cmap='coolwarm', mask=mask) # https://www.mathworks.com/matlabcentral/answers/652698-number-of-digits-in-heatmap-plot

plt.title('Correlation Matrix [Physically Collected]') 
plt.savefig('ChlaHMCP_SH.png', bbox_inches = 'tight', pad_inches=1, dpi = 450)
plt.show()
