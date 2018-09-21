# -*- coding: utf-8 -*-

'''
seq2seq用にtext8コーパスから学習データと正解データを作成

'''

from __future__ import print_function

import numpy as np
import re
import sys
import datetime
import os
import os.path
import subprocess

#----- グローバル変数一覧 -----
my_epoch=100
vec_size=100
maxlen_words = 5
KeyError_set=set()
today_str=''
tmp_vec_dict=dict()


#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today



#listの各要素を単語で連結してstring型で返す
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#文の長さの乱数
def rand_sent():
    li=list(range(1,41))
    we=[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03135889, 0.06271777, 0.06271777, 0.09756098, 0.1358885, 0.09407666, 0.08710801, 0.10801394, 0.08362369, 0.04181185, 0.05574913, 0.04529617, 0.05574913, 0.03832753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    we[15]=1.0-sum(we)+we[15]


    return np.random.choice(li, p=we)


#空所の長さの乱数
def rand_cloze():
    li=list(range(1,6))
    we=[0.615502686109, 0.235610130468, 0.114351496546, 0.0283960092095, 0.00613967766692]
    we[0]=1.0-sum(we)+we[0]
    return np.random.choice(li, p=we)


#前処理
def preprocess_line2(before_line):
    after_line=before_line.lower()
    after_line = re.sub(r'[^a-z{}]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)
    after_line = after_line.strip()

    return after_line



def make_data(old_path, cloze_path, ans_path):
    if (os.path.exists(ans_path)==False):
        print('make data start...')
        i=0
        with open(old_path) as f_in:
            with open(cloze_path, 'w') as f_cloze:
                with open(ans_path, 'w') as f_ans:
                    for line in f_in:
                        i+=1
                        print('line: '+str(i)+'\n')
                        line=preprocess_line2(line)
                        line_list=line.split(' ')
                        line_len=len(line_list)
                        if (line_len > 8) and (line_len) < 30:
                            #cloze_lenは1～5のどれか
                            cloze_len=rand_cloze()
                            #cloze_startは0～
                            #例えば1行に10単語あって，空所が2単語なら，
                            #randint(9)となり，cloze_startは0～8
                            cloze_start=np.random.randint(line_len-cloze_len+1)
                            cloze_end=cloze_start+cloze_len+1

                            line_list.insert(cloze_start, '{')
                            line_list.insert(cloze_end, '}')
                            ans_text=' '.join(line_list)
                            cloze_text=re.sub('\{.+\}', '{ }',ans_text)

                            f_ans.write(ans_text+'\n')
                            f_cloze.write(cloze_text+'\n')

        print_time('make data end')




#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
tmp_path='text8_nmt_tmp.txt'

print('Loading  '+tmp_path)
file_name=tmp_path[:-4]
make_data(tmp_path, file_name+'_nmt_cloze.txt', file_name+'_nmt_ans.txt')
