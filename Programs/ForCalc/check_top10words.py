# -*- coding: utf-8 -*-

'''
空所に入る語の分布確認
'''

from __future__ import print_function

import collections
import math
import os
import re
import subprocess



#----- 自作関数 -----

def is_correct_cloze(line):
    left=line.count('{')
    right=line.count('}')
    if left*right==1:
        return True
    else:
        #print(line)
        pass
    return False



def print_top10words(file_path):
    words_d=dict()
    with open(file_path) as f:
        for line in f:
            line=re.sub(r'[^a-z{}<> ]', '', line)
            if is_correct_cloze(line):
                line=re.sub(r'.*{ ', '', line)
                line_cloze=re.sub(r' }.*', '', line)
                words=line_cloze.split(' ')
                len_words=len(words)
                if len_words==1:
                    w=words[0]
                    if not w=='':
                        if w in words_d:
                            words_d[w]+=1
                        else:
                            words_d[w]=1
                elif len_words>1:
                    for x in words:
                        if x in words_d:
                            words_d[x]+=1
                        else:
                            words_d[x]=1

    words_list = sorted(words_d.items(), key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(words_list[i])
    print(len(words_list))



def print_result(file_path):
    print('\n\nfile: ',file_path)
    print_top10words(file_path)




#return なし


#----- いわゆるmain部みたいなの -----
'''
print_result('/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt.ans')
print_result('/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt_dev.ans')
print_result('output_dev.txt')
print_result('/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/center_nmt.ans')
print_result('output_infer.txt')
'''
print_result('/home/tamaki/M2/pytorch_data/text8_ans.txt')
