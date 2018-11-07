# -*- coding: utf-8 -*-

'''
学習データの未知語をUNKに置換するやつ
kenLM用

'''

from __future__ import print_function

import numpy as np
import re
import sys
import datetime
import os
import os.path

#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today

def get_words(file):
    words=[]
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=normalizeString(line)
            for word in line.split(' '):
                if not word in words:
                    words.append(word)

    return words



def change_unk(input_data, output_data, vocab):
    with open(input_data, encoding='utf-8') as f_in:
        with open(output_data, 'w') as f_out:
            i=0
            ct=0
            #TODO text8コーパスは1行だからこの書き方だけど，それ以外ならiの初期化とか必要そう
            for line in f_in:
                line=line.strip()
                #text8コーパスなので前処理は考慮してない
                words=line.split(' ')
                for word in words:
                    if not word in vocab:
                        word='#UNK#'
                        ct+=1
                    if i==0:
                        f_out.write(word)
                        i+=1
                    else:
                        f_out.write(' '+word)
                        i+=1
    print('word num   : ', i)
    print('change UNK : ', ct)



#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')


#データ
tmp_path='../../../'
input_data=tmp_path+'text8.txt'
output_data=tmp_path+'text8_UNK.txt'

vocab_path=tmp_path+'enwiki_vocab30000_wordonly.txt'

all_words = get_words(vocab_path)

change_unk(input_data, output_data, all_words)
