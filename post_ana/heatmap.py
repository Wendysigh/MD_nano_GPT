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

def load_xvg(path):
    phi, psi = [], []
    with open(path) as f:
        for line in f:
            if line.startswith('#') or line.startswith('@'):
                continue

            cols = line.split()
            phi.append(float(cols[1]))
            psi.append(float(cols[0]))
    return (phi,psi)

for i in range(0,100):
    if i<10:
        url = f'/home/wzengad/projects/MD_code/data/phi_psi_raw/ala2-0.1ps-0{i}.xvg'
    else:
        url = f'/home/wzengad/projects/MD_code/data/phi_psi_raw/ala2-0.1ps-{i}.xvg'
    # data = np.loadtxt(url,comments=["@", '#'],unpack=True)
    (_phi, _psi) = load_xvg(url)

    phi=np.append(phi,_phi)
    psi=np.append(psi,_psi)
    
bins=np.arange(-180, 180, 1.8)

# idx_sin_phi = np.loadtxt('/home/wzengad/projects/MD_code/data/phi/train',dtype=int).reshape(-1)
# idx_sin_psi = np.loadtxt('/home/wzengad/projects/MD_code/data/psi/train',dtype=int).reshape(-1)
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

# import seaborn as sns
# #%matplotlib inline
# sns.set(font_scale=1.5)
# #sns.set_context({"figure.figsize":(8,8)})
# import matplotlib.pyplot as plt
# ax=sns.heatmap(data=data[::-1,:],square=True,cmap="RdBu_r",alpha = 0.7)
# # ax=sns.heatmap(data=data[::-1,:],square=True,cmap="RdBu_r",alpha = 0.7,yticklabels=yticklabels, xticklabels=xticklabels)
# # ax=sns.heatmap(data=data.T[:,::-1],square=True,cmap="RdBu_r",alpha = 0.9,yticklabels=yticklabels)
# # ax=sns.heatmap(data=data.T[:,::-1],cmap="RdYlGn",alpha = 0.8,yticklabels=10,xticklabels=10) 
# # ax.set_yticklabels(yticklabels)
# # ax.set_xticklabels(xticklabels)
# ax.set_yticks([0,50,100,150])
# ax.set_xticks([0,50,100,150])
# # plt.show()

# fig = ax.get_figure()
# fig.savefig('hist.png')
# ax.clear()

import matplotlib.style as style 
import matplotlib
import matplotlib.pyplot as plt
style.available
style.use('seaborn-paper') #sets the size of the charts
# style.use('ggplot')
matplotlib.rcParams['font.family'] = "serif"
yticklabels = [180, 90, 0, -90]
xticklabels = [-180, -90, 0, 90]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_yticks([0,50,100,150])
ax.set_yticklabels(yticklabels, rotation=90)
ax.set_xticks([0,50,100,150])
ax.set_xticklabels(xticklabels,)

im = ax.imshow(data[::-1,:], cmap='YlGn')
plt.colorbar(im)
# plt.title("This is a title")
plt.savefig('ala_heat.pdf', format='pdf', dpi=600, pad_inches = 0.05)

