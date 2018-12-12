# -*- coding: utf-8 -*-
'''
英文平均何単語か調べるやつ
'''

import numpy as np
import glob


tlist=np.zeros(100)


tmp_path='../../../pytorch_data/tmpCLOTH/CLOTH'

file1 = sorted(glob.glob(tmp_path + '*ans.txt'))
for file_name in file1:
    print(file_name)
    word=0
    sent=0
    max=0
    min=100

    #with open('../../Data/my_nmt/center_ans.txt') as f:
    with open(file_name) as f:
        for line in f:
            tmp=line.strip().split(' ')
            tmp=line.split(' ')
            slen=len(tmp)-2 #{}の文引く
            if max < slen:
                max=slen
            if min > slen:
                min = slen
            if slen<5:
                print(file_name)
                print(line)
                #exit()
            word+=slen
            sent+=1
            #tlist[slen-1]+=1
    '''
    n=0
    for x in tlist:
        if x <= 5:
            tlist[n]=0.0
        n+=1

    n=0
    sum2=0
    snum=0
    for x in tlist:
        n+=1
        if x > 0:
            tmp1=x
            snum+=tmp1
            tmp1=n*tmp1
            sum2+=tmp1
    '''



    print('ave',word/sent)
    print('max', max)
    print('min',min)
    '''
    print(tlist)
    sum1=sum(tlist)
    print(tlist/sum1)
    print(1.0*sum2/snum)
    '''
