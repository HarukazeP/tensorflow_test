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


#学習データへの前処理を行う
#小文字化，アルファベット以外の文字の削除，1万単語ごとに分割
def parse_line(old_path, new_path):
    if (os.path.exists(new_path)==False):
        print('Preprpcessing training data...')
        text=''
        text_len=0
        i=0
        with open(old_path) as f_in:
            with open(new_path, 'w') as f_out:
                for line in f_in:
                    #この前処理はtext8とかの前処理と同じ
                    line=line.strip()
                    line_list=line.split(' ')
                    line_len=len(line_list)
                    #max_len以下の時は連結して次へ
                    max_len=rand_sent()
                    if(text_len+line_len <= max_len):
                        if(text_len==0):
                            text=line
                        else:
                            text=text+' '+line
                        text_len=text_len+line_len
                    #max_lenより長いときはmax_len単語ごとに区切ってファイルへ書き込み
                    else:
                        while (line_len>max_len):
                            if(text_len==0):
                                text=list_to_sent(line_list,0,max_len)
                            else:
                                text=text+' '+list_to_sent(line_list,0,max_len-text_len)
                            f_out.write(text+'\n')
                            text=''
                            text_len=0
                            #残りの更新
                            line_list=line_list[max_len-text_len+1:]
                            line_len=len(line_list)
                            max_len=rand_sent()
                        #while 終わり（1行の末尾の処理）
                        #余りは次の行と連結
                        text=list_to_sent(line_list,0,line_len)
                        text_len=line_len
                #for終わり（ファイルの最後の行の処理）
                if text_len!=0:
                    text=preprocess_line2(text)
                    f_out.write(text+'\n')
                print('total '+str(i)+' line\n')
                print_time('preprpcess end')


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
#tmp_path='text8.txt'
file_path='/media/tamaki/HDCL-UT/tamaki/M2/data_for_kenlm/enwiki1GB.txt'
output_path='/media/tamaki/HDCL-UT/tamaki/M2/data_for_seq2seq/enwiki1GB'
print('Loading  '+file_path)
tmp_path=output_path+'_seq2seq.txt'

parse_line(file_path, tmp_path)
make_data(tmp_path, output_path+'_seq2seq_cloze.txt', file_name+'_seq2seq_ans.txt')
