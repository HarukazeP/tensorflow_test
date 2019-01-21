# -*- coding: utf-8 -*-

'''
CLOTH_ansからKenLM用のデータ作成

'''

from __future__ import print_function

import re
import datetime

import json
from collections import OrderedDict
import nltk
import unicodedata


#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today




#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def preprocess(s):
    sent_tokens=[]
    s = unicodeToAscii(s)
    s = re.sub(r'[ ]+', ' ', s)
    s = s.strip()
    tokens=nltk.word_tokenize(s)
    symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
    num_tag=("LS", "CD")
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if tag in symbol_tag:
            pass
            #記号は無視
        elif tag in num_tag:
            sent_tokens.append('NUM')
        else:
            sent_tokens.append(word.lower())

    return ' '.join(sent_tokens)


#空所記号の削除
def make_sents(ans_sent):
    ans_sent=re.sub(r'{ ', '', ans_sent)
    ans_sent=re.sub(r' }', '', ans_sent)
    ans_sent=ans_sent.strip()
    sent=preprocess(ans_sent)

    return sent


#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
tmp_path='../../../pytorch_data/CLOTH_for_model/'
train_ans=tmp_path+'CLOTH_valid_ans.txt'
i=0
with open(train_ans, encoding='utf-8') as f:
    with open(tmp_path+'for_KenLM_CLOTH_val.txt', 'w') as f_out:
        for line in f:
            i+=1
            if(i%1000==0):
                print('line:',i)
            out=make_sents(line)
            f_out.write(out+'\n')
