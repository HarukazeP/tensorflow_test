# -*- coding: utf-8 -*-
'''
空所内に何単語あるか調べるやつ
'''


len1=0
len2=0
len3=0
len4=0
len5=0
len6=0

import glob
import nltk

#tmp_path='../../Data/CLOTH'
tmp_path='../../../pytorch_data/CLOTH_for_model/CLOTH_'

file1 = sorted(glob.glob(tmp_path + '*ans.txt'))
for file_name in file1:
    #with open('../../Data/my_nmt/center_choices.txt') as f:
    with open(file_name) as f:
        print(file_name)
        for line in f:
            line=line.strip()
            if line.count('>')>0:
                print(line)
            line=line[line.find('{')+2:line.rfind('}')-2]
            tmp=line.split(' ### ')
            for x in tmp:
                tokens=nltk.word_tokenize(x.strip())
                xlen=len(tokens)
                if xlen==1:
                    len1+=1
                elif xlen==2:
                    len2+=1
                elif xlen==3:
                    len3+=1
                elif xlen==4:
                    len4+=1
                    print(x)
                    print(tokens)
                elif xlen==5:
                    len5+=1
                    print(x)
                    print(tokens)
                elif xlen==6:
                    len6+=1
                else:
                    #print(x)
                    pass

    sum=len1+len2+len3+len4+len5

    print('1:',len1)
    print('2:',len2)
    print('3:',len3)
    print('4:',len4)
    print('5:',len5)
    print('6:',len6)
    '''
    print(1.0*len1/sum)
    print(1.0*len2/sum)
    print(1.0*len3/sum)
    print(1.0*len4/sum)
    print(1.0*len5/sum)
    '''
