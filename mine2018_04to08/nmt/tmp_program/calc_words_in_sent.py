# -*- coding: utf-8 -*-
'''
センターの英文平均何単語か調べるやつ
'''

import numpy as np

word=0
sent=0
max=0
min=40

tlist=np.zeros(40)

with open('../../Data/my_nmt/center_ans.txt') as f:
    for line in f:
        tmp=line.split(' ')
        slen=len(tmp)-2
        if max < slen:
            max=slen
        if min > slen:
            min = slen
        word+=slen
        sent+=1
        tlist[slen-1]+=1
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



print(1.0*word/sent)
print(max)
print(min)
print(tlist)
sum1=sum(tlist)
print(tlist/sum1)
print(1.0*sum2/snum)

