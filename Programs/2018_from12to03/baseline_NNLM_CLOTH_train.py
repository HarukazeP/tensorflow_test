# -*- coding: utf-8 -*-

'''
ベースライン用のNNLM
seq2seqの時はpytorchで書いてたけど、kerasで書きなおしたやつ
構造は同じはず

CLOTHで学習、CLOTHと同じ前処理

#TODO まだ未作成、まだ何も手つけてない

動かしていたバージョン
python  : 3.5.2

'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import datetime

import time
import math

import numpy as np
import os
import argparse
import collections
import kenlm

import nltk

#----- グローバル変数一覧 -----

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'

git_data_path='../../Data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')

#----- 関数群 -----



#空所つき英文読み取り
def readCloze(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())

    return data


#選択肢読み取り
def readChoices(file_name):
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            line=line.strip()
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


#空所内の単語読み取り
def readAns(file_name):
    data=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            data.append(line.strip())

    return data

#正答一つをidsへ
def ans_to_ids2(ans, choices):
    # ans は文字列
    # choices は文字列のリスト
    ids = [1 if choi==ans else 0 for choi in choices]

    #デバッグ時確認用
    if sum(ids)!=1:
        print('### ans_to_ids ERROR')
        print(ids)
        print(ans)
        print(choices)
        exit()

    return ids


class Ngram2():
    def __init__(self, model):
        self.model=model

    #前処理
    def preprocess(self, s):
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

        return sent_tokens

    #空所に選択肢を補充した4文を生成
    def make_sents(self, cloze_sent, choices):
        sents=[]
        before=re.sub(r'{.*', '', cloze_sent)
        after=re.sub(r'.*}', '', cloze_sent)
        for choice in choices:
            tmp=before + choice + after
            tmp=tmp.strip()
            sents.append(tmp)
        return sents


    def sent_to_KenLM_score2(self, sent):
        tokens=self.preprocess(sent)
        kenlm_sent=' '.join(tokens)
        sent_len=len(tokens)

        KenLM_score=1.0*self.model.score(kenlm_sent)/sent_len

        return KenLM_score


    def get_KenLM_score2(self, cloze_list, choices_list):
        KenLM_score=[]
        for cloze_sent, choices in zip(cloze_list, choices_list):
            s1, s2, s3, s4=self.make_sents(cloze_sent, choices)
            s1_score = self.sent_to_KenLM_score2(s1)
            s2_score = self.sent_to_KenLM_score2(s2)
            s3_score = self.sent_to_KenLM_score2(s3)
            s4_score = self.sent_to_KenLM_score2(s4)
            KenLM_score.append([s1_score, s2_score, s3_score, s4_score])

        #対数頻度，log1p(x)はlog_{e}(x+1)を返す
        #頻度が0だとまずいので念のため+1してる
        KenLM_array=np.array(KenLM_score, dtype=np.float)

        return KenLM_array



#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def model_test(model_path, cloze_path, choices_path, ans_path, data_name=''):
    print(data_name)

    test_X=readCloze(cloze_path)
    test_C=readChoices(choices_path)
    test_Y=readAns(ans_path)

    Y_test=np.array([ans_to_ids2(s, c) for s,c in zip(test_Y, test_C)], dtype=np.bool)
    model = kenlm.LanguageModel(model_path)

    Y_pred=Ngram2(model).get_KenLM_score2(test_X, test_C)
    OK=0
    line=0

    for p,y in zip(Y_pred, Y_test):
        line+=1
        if p.argmax()==y.argmax():
            OK+=1

    print('acc:', 1.0*OK/line)


#----- main部 -----
if __name__ == '__main__':

    center_cloze=git_data_path+'center_cloze.txt'
    center_choi=git_data_path+'center_choices.txt'
    center_ans=git_data_path+'center_ans.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_choi=git_data_path+'microsoft_choices_for_CLOTH.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'

    high_path=git_data_path+'CLOTH_test_high'
    middle_path=git_data_path+'CLOTH_test_middle'

    CLOTH_high_cloze = high_path+'_cloze.txt'
    CLOTH_high_choi = high_path+'_choices.txt'
    CLOTH_high_ans = high_path+'_ans.txt'

    CLOTH_middle_cloze = middle_path+'_cloze.txt'
    CLOTH_middle_choi = middle_path+'_choices.txt'
    CLOTH_middle_ans = middle_path+'_ans.txt'

    model_test(model, center_cloze, center_choi, center_ans, data_name='center')

    model_test(model, CLOTH_high_cloze, CLOTH_high_choi, CLOTH_high_ans, data_name='CLOTH_high')

    model_test(model, CLOTH_middle_cloze, CLOTH_middle_choi, CLOTH_middle_ans, data_name='CLOTH_middle')
