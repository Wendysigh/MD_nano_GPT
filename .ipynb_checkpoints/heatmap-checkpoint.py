from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
tf.compat.v1.enable_eager_execution() 
import numpy as np
import os
from bs4 import BeautifulSoup
import requests
import json
import time
# from lossT import sparse_categorical_crossentropy

phi=np.array([])
psi=np.array([])


for i in range(0,100):
    if i<10:
        url = 'https://chz276.ust.hk/users/visitor/rama-traj/ala2-0.1ps-0'+str(i)+'.txt'
    else:
        url = 'https://chz276.ust.hk/users/visitor/rama-traj/ala2-0.1ps-'+str(i)+'.txt'
    link = requests.get(url)
    soup_link = BeautifulSoup(link.content, 'html')
    
    t1=soup_link.get_text().split('\n')
    t1.remove('')
    t2=[i.split(' ')for i in t1 ]
    
    _phi=np.array([float(i[0]) for i in t2[:100000]])
    _psi=np.array([float(i[1]) for i in t2[:100000]])
    
    phi=np.append(phi,_phi)
    psi=np.append(psi,_psi)
    
bins=np.arange(-180, 180, 9)
idx_sin_phi=np.digitize(phi, bins) 
idx_sin_psi=np.digitize(psi, bins) 

import pandas as pd
count=pd.DataFrame({'phi':idx_sin_phi,'psi':idx_sin_psi})
cnt=count.groupby(['phi','psi'])

test=cnt.agg(len)
data=np.zeros((len(bins),len(bins)))

import math
for i in range(len(bins)):
    for j in range((len(bins))):
        if (i,j) in test:
            data[i,j]=math.log10(test[(i,j)])   #i:phi,j:psi
data2=pd.DataFrame(data)

import seaborn as sns
%matplotlib inline
sns.set(font_scale=1.5)

yticklabels = [-bins[idx] for idx in [0,10,20,30]][::-1]
xticklabels = [bins[idx] for idx in [0,10,20,30]]

#sns.set_context({"figure.figsize":(8,8)})
import matplotlib.pyplot as plt
#ax=sns.heatmap(data=data.T[:,::-1],square=True,cmap="coolwarm",alpha = 0.9,yticklabels=yticklabels)
ax=sns.heatmap(data=data.T[:,::-1],cmap="RdYlGn",alpha = 0.8,yticklabels=10,xticklabels=10) 
ax.set_yticklabels(yticklabels)
ax.set_xticklabels(xticklabels)

plt.show()

fig = ax.get_figure()
fig.savefig('hist.png')