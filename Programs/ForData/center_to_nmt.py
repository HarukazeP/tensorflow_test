# -*- coding: utf-8 -*-

'''
センターをtext8コーパスと同様の前処理

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

#前処理
def preprocess_line2(before_line):
    after_line=before_line.lower()
    after_line=after_line.replace('0', ' zero ')
    after_line=after_line.replace('1', ' one ')
    after_line=after_line.replace('2', ' two ')
    after_line=after_line.replace('3', ' three ')
    after_line=after_line.replace('4', ' four ')
    after_line=after_line.replace('5', ' five ')
    after_line=after_line.replace('6', ' six ')
    after_line=after_line.replace('7', ' seven ')
    after_line=after_line.replace('8', ' eight ')
    after_line=after_line.replace('9', ' nine ')
    after_line = re.sub(r'[^a-z{}]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)
    after_line = after_line.strip()

    return after_line

#前処理して新規ファイル作成
def clean_line(old_path, new_path):
    print('cleaning data...')
    with open(old_path) as f_in:
        with open(new_path, 'w') as f_out:
            for line in f_in:
                #この前処理はtext8とかの前処理と同じ
                line=preprocess_line2(line)
                f_out.write(line)
    print('clean end')








#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')


#データ
n100_path='../../Data/my_nmt/center_ans.txt'
n500_path='../../Data/my_nmt/center_cloze.txt'

clean_line(n100_path, '../../Data/my_nmt/center_nmt.ans')
clean_line(n500_path, '../../Data/my_nmt/center_nmt.cloze')
