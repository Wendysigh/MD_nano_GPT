import numpy as np
import os
import time
import requests
import json
import pandas as pd
import argparse
import shutil
from tqdm import tqdm

# For Fip35-macro

x=np.array([])
total_num = 57
train = np.array([])
valid = np.array([])
train_num = int(0.8 * total_num)

# choice = 'macro5'
choice = 'micro1000'

for i in range(train_num):
    if choice=='macro5':
        path = f"data/Fip35/unzip_all/regen/{choice}/macro5_{i}"
    else:
        path = f"data/Fip35/unzip_all/regen/{choice}/traj_{i}"
    # path = f"data/Fip35/macro5/macro5_{i}"
    single_file = np.loadtxt(path,dtype=int)
    if single_file.shape[0] % 5 != 0:
        single_file = single_file[:-(single_file.shape[0] % 5)]
    print(single_file.shape)
    # if i == 23:
    #     single_file = single_file[:32585] # continue # 23 has entry, 32586, which cannot be subsampled by 5
    train = np.append(train, single_file)

for i in range(train_num, total_num):
    if choice=='macro5':
        path = f"data/Fip35/unzip_all/{choice}/macro5_{i}"
    else:
        path = f"data/Fip35/unzip_all/regen/{choice}/traj_{i}"
    single_file = np.loadtxt( path , dtype=int)
    print(single_file.shape)
    valid = np.append(valid, single_file)

print(train.shape)
print(valid.shape)
np.savetxt(f'data/Fip35_{choice}/train',train,fmt='%i')
np.savetxt(f'data/Fip35_{choice}/test',valid,fmt='%i')



# For Fip35-micro

x=np.array([])
total_num = 55 
train = np.array([])
valid = np.array([])
train_num = int(0.8 * total_num)

for i in range(train_num):
    single_file = np.loadtxt(f'data/Fip35/micro1000/traj_{i}',dtype=int)
    train = np.append(train, single_file)
for i in range(train_num, total_num):
    single_file = np.loadtxt(f'data/Fip35/micro1000/traj_{i}',dtype=int)
    valid = np.append(valid, single_file)
np.savetxt('data/Fip35_micro/train',train,fmt='%i')
np.savetxt('data/Fip35_micro/test',valid,fmt='%i')


# For Macro
from bs4 import BeautifulSoup
x=np.array([])
for i in range(100):
    if i <10:
        single_file = np.loadtxt(f'/home/wzengad/projects/MD_code/data/MacroAssignment/ala2-0.1ps-0{i}_macro.txt',dtype=int)
    else:
        single_file = np.loadtxt(f'/home/wzengad/projects/MD_code/data/MacroAssignment/ala2-0.1ps-{i}_macro.txt',dtype=int)

    x = np.append(x, single_file)
num = int(0.8 * len(x))    
train = x[:num]
valid = x[num:]
np.savetxt('/home/wzengad/projects/MD_code/data/MacroAssignment/train',train,fmt='%i')
np.savetxt('/home/wzengad/projects/MD_code/data/MacroAssignment/test',valid,fmt='%i')

np.random.seed(0)
train_id = np.random.choice(100,80,replace=False)
test_id = set(range(100)) - set(train_id)

x=np.array([])
test=np.array([])
train = np.loadtxt('/home/wzengad/projects/MD_code/data/RMSD/train',dtype=int)
train2 = train.reshape(-1,80)
valid = np.loadtxt('/home/wzengad/projects/MD_code/data/RMSD/test',dtype=int)
valid2 = valid.reshape(-1,20)
all = np.hstack((train2, valid2))

new_train = all[:,train_id]
new_test = all[:,list(test_id)]

np.savetxt('/home/wzengad/projects/MD_code/data/RMSD/train_random',new_train,fmt='%i')
np.savetxt('/home/wzengad/projects/MD_code/data/RMSD/test_random',new_test,fmt='%i')

for i in train_id: 
    if i<10:
        url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-0'+str(i)+'_assignment_.txt'
    else:
        url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-'+str(i)+'_assignment_.txt'
    link = requests.get(url)

    soup_link = BeautifulSoup(link.content, 'html')
    t1=soup_link.get_text().split('\n')
    t1.remove('')
    t2=np.array(t1).astype(int)

    x=np.append(x,t2)

np.savetxt('/home/wzengad/projects/MD_code/data/RMSD/train',x,fmt='%i')

for i in test_id: 
    if i<10:
        url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-0'+str(i)+'_assignment_.txt'
    else:
        url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-'+str(i)+'_assignment_.txt'
    link = requests.get(url)

    soup_link = BeautifulSoup(link.content, 'html')
    t1=soup_link.get_text().split('\n')
    t1.remove('')
    t2=np.array(t1).astype(int)

    test=np.append(test,t2)

np.savetxt('/home/wzengad/projects/MD_code/data/RMSD/test',test,fmt='%i')



if data=='macro':    
    savedir=f"data/de_recrossing/"
    os.makedirs(savedir, exist_ok=True)
    for step in [1,2,5,10,20,50]:
        input_x=np.array([])
        valid_x=np.array([])
        for i in range(0,100): 
            if i<10:
                url = f"https://chz276.ust.hk/users/visitor/ala2_0.1ps_de_recrossing/{step}/ala2-0.1ps-0{i}_macro.txt"
            else:
                url = f"https://chz276.ust.hk/users/visitor/ala2_0.1ps_de_recrossing/{step}/ala2-0.1ps-{i}_macro.txt"
            link = requests.get(url)
            soup_link = BeautifulSoup(link.content, 'html')
            t1=soup_link.get_text().split('\n')
            t1.remove('')
            t2=np.array(t1).astype(int)
            #input_x=np.append(input_x,t2[:50000].reshape(5000,10).T.flatten())
            input_x=np.append(input_x,t2[:int(0.8*len(t2))])
            valid_x=np.append(valid_x,t2[int(0.8*len(t2)):])

        np.savetxt(savedir+'{}_train'.format(step),input_x,fmt='%i')
        np.savetxt(savedir+'{}_valid'.format(step),valid_x,fmt='%i')
elif data=='2d':
    input_x=np.array([])
    valid_x=np.array([])

    url = 'https://chz276.ust.hk/users/visitor/2D_potential_traj_900states.txt'
    link = requests.get(url)
    soup_link = BeautifulSoup(link.content, 'html')

    t1=soup_link.get_text().split('\n')
    t1.remove('')
    t2=np.array([int(i.split()[0]) for i in t1]).astype(int)

    input_x=t2[:int(0.5*len(t2))]
    valid_x=t2[int(0.5*len(t2)):]
elif data=='subsample_alanine':
    input_x=np.array([])
    valid_x=np.array([])
    md=np.array([])
    for i in range(0,100): 
        if i<10:
            url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-0'+str(i)+'_assignment_.txt'
        else:
            url = 'https://chz276.ust.hk/users/visitor/ala2_Microassignment/ala2-0.1ps-'+str(i)+'_assignment_.txt'
        link = requests.get(url)

        soup_link = BeautifulSoup(link.content, 'html')
        t1=soup_link.get_text().split('\n')
        t1.remove('')
        t2=np.array(t1).astype(int)
        input_x=np.append(input_x,t2[:50000].reshape(5000,10).T.flatten())
        #input_x=np.append(input_x,t2[:50000])
        #valid_x=np.append(valid_x,t2[50000:])
        valid_x=np.append(valid_x,t2[50000:100000].reshape(5000,10).T.flatten())
        md=t2[:100000].reshape(10000,10).T.flatten()
        np.savetxt(save_dir+'md_'+str(i),md,fmt='%i')
        
elif data=='state_length':
    t,s,l=np.array([]),np.array([]),np.array([])
    t_v,s_v,l_v=np.array([]),np.array([]),np.array([])
    #md=np.array([])
    for i in range(0,100): 
        if i<10:
            url =f"https://chz276.ust.hk/users/siqin/workspace/Wenqi/ala2_1ps_MD_test/SparseMacroAssignment/ala2-0.1ps-0{i}_macro.txt"
        else:
            url =f"https://chz276.ust.hk/users/siqin/workspace/Wenqi/ala2_1ps_MD_test/SparseMacroAssignment/ala2-0.1ps-{i}_macro.txt"
        link = requests.get(url)

        soup_link = BeautifulSoup(link.content, 'html')
        t2=soup_link.get_text().split('\n')
        t2.remove('')
        t1=np.array(t2[:int(0.8*len(t2))])
        t1_v=np.array(t2[int(0.8*len(t2)):])


        state=np.array([int(t1[i].split(' ')[0]) for i in range(len(t1))])
        length=np.array([int(t1[i].split(' ')[1]) for i in range(len(t1))])
        state_v=np.array([int(t1_v[i].split(' ')[0]) for i in range(len(t1_v))])
        length_v=np.array([int(t1_v[i].split(' ')[1]) for i in range(len(t1_v))])



        t=np.append(t,t1)
        s=np.append(s,state)
        l=np.append(l,length)

        t_v=np.append(t_v,t1_v)
        s_v=np.append(s_v,state_v)
        l_v=np.append(l_v,length_v)


    d={'raw':t,'state':s.astype(int),'length':l.astype(int)}
    train=pd.DataFrame(d)

    d_v={'raw':t_v,'state':s_v.astype(int),'length':l_v.astype(int)}
    valid=pd.DataFrame(d_v)
    train.to_csv(r'data/state_length_macro/train.txt')
    valid.to_csv(r'data/state_length_macro/valid.txt')
    
    # macro https://chz276.ust.hk/users/visitor/ala2-4state-assignments/MacroAssignment/ala2-0.1ps-01_macro.txt