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



#TODO
#まだ全然作ってない
def csv_to_text(test_csv, ans_csv, output_pass):
    cloze_file = output_pass+'microsoft_cloze.txt'
    ans_file = output_pass+'microsoft_ans.txt'
    choice_file = output_pass+'microsoft_choice.txt'

    test_df=pd.read_csv(test_csv)
    ans_df=pd.read_csv(ans_csv)

    with open(cloze_file, 'w') as f_clz:
        with open(ans_file, 'w') as f_ans:
            with open(choice_file, 'w') as f_cho:
                #csvの各行に対してループ
                for i in range(len(test_df)):

                    sent=test_df.ix[i, 'question']
                    ans_id=ans_df.ix[i, 'answer']
                    ans_word=test_df.ix[i, ans_id+')']

                    cloze, ans, choice=make_test_sents(sent, choices, ans_word)
                    f_clz.write(cloze+'\n')
                    f_ans.write(ans+'\n')
                    f_cho.write(choice+'\n')



#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
tmp_path='../../../'
test_data=tmp_path+'testing_data.csv'
ans_data=tmp_path+'test_answer.csv'

csv_to_text(test_data, ans_data, tmp_path)
