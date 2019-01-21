# -*- coding: utf-8 -*-

'''
microsoft社の英文空所補充問題を4択に帰る
MPNetのテスト用

'''

from __future__ import print_function

import numpy as np
import re
import sys
import datetime
import os
import os.path

import random

#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today


def make_new_choices_line(choices, ans):
    try:
        ans_index=choices.index(ans)
    except ValueError:
        print(choices)
        print(ans)
        exit()

    remove_index=random.randint(0,4)
    while(ans_index==remove_index):
        remove_index=random.randint(0,4)
    choices.pop(remove_index)

    output=' ### '.join(choices)
    output='{ '+output+' }'

    return output





def make_new_choices_file(choices_path, ans_path, out_path):

    with open(choices_path, encoding='utf-8') as f_c:
        with open(ans_path, encoding='utf-8') as f_a:
            with open(out_path, 'w') as f_o:
                for c_line, ans_line in zip(f_c, f_a):
                    before=re.sub(r'{.*', '', c_line)
                    before=re.sub(r'\n', '', before)
                    #before=before.strip()

                    after=re.sub(r'.*}', '', c_line)
                    after=re.sub(r'\n', '', after)


                    ans=re.sub(r'.*{ ', '', ans_line)
                    ans=re.sub(r' }.*', '', ans)
                    ans=ans.strip()

                    choices=re.sub(r'.*{ ', '', c_line)
                    choices=re.sub(r' }.*', '', choices)
                    choices=choices.strip()

                    c_list=choices.split(' ### ')     #選択肢を区切る文字列

                    new_choices=make_new_choices_line(c_list, ans)

                    new_line=before+new_choices+after
                    f_o.write(new_line+'\n')





#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

git_data_path='../../Data/'

MS_choi=git_data_path+'microsoft_choices.txt'
MS_ans=git_data_path+'microsoft_ans.txt'

out_MS_choi=git_data_path+'microsoft_choices_for_CLOTH.txt'

make_new_choices_file(MS_choi, MS_ans, out_MS_choi)







pass
