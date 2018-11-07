# -*- coding: utf-8 -*-

'''
kenLMライブラリを用いた統計言語モデル（n-grams）
https://github.com/kpu/kenlm

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



#----- グローバル変数一覧 -----
MAX_LENGTH = 40
HIDDEN_DIM = 128
EMB_DIM = 100
BATCH_SIZE = 128

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
git_data_path='../../Data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + '/' + today_str
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3

#事前処理いろいろ

#----- 関数群 -----



#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


#データの前処理
#strip()は文頭文末の改行や空白を取り除いてくれる
def normalizeString(s, choices=False):
    s = unicodeToAscii(s.lower().strip())
    #text8コーパスと同等の前処理
    s=s.replace('0', ' zero ')
    s=s.replace('1', ' one ')
    s=s.replace('2', ' two ')
    s=s.replace('3', ' three ')
    s=s.replace('4', ' four ')
    s=s.replace('5', ' five ')
    s=s.replace('6', ' six ')
    s=s.replace('7', ' seven ')
    s=s.replace('8', ' eight ')
    s=s.replace('9', ' nine ')
    if choices:
        s = re.sub(r'[^a-z{}#]', ' ', s)
    else:
        s = re.sub(r'[^a-z{}]', ' ', s)
    s = re.sub(r'[ ]+', ' ', s)

    return s.strip()


#入出力データ読み込み用
def readData(input_file, target_file):
    #print("Reading data...")
    pairs=[]
    i=0
    with open(input_file, encoding='utf-8') as input:
        with open(target_file, encoding='utf-8') as target:
            for line1, line2 in zip(input, target):
                i+=1
                pairs.append([normalizeString(line1), normalizeString(line2)])
    print("data: %s" % i)

    return pairs


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



#ペアじゃなくて単独で読み取るやつ
def readData2(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(normalizeString(line))

    return data

def get_choices(file_name):
    print("Reading data...")
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=get_cloze(normalizeString(line, choices=True))
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


def get_cloze(line):
    line=re.sub(r'.*{ ', '', line)
    line=re.sub(r' }.*', '', line)

    return line


#選択肢を補充した文4つを返す
def make_sents(choices, cloze_sent):
    sents=[]
    before=re.sub(r'{.*', '', cloze_sent)
    after=re.sub(r'.*}', '', cloze_sent)
    for choice in choices:
        tmp=before + choice + after
        sents.append(tmp.strip())

    return sents



#選択肢が全て1語のデータのみについて
#選択肢補充した文と，その正答のペアを返却
def make_data(data_pair, choices_lists, one_word=True):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        if(one_word==True):
            for choice in choices:
                if(len(choice.split(' '))>1):
                    flag=-1
                    #選択肢に2語以上のものがあるときはflagが負
            if(flag>0):
                test_data=make_sents(choices, sent[0])
                test_data.append(sent[1])
                data.append(test_data)
        else:
            test_data=make_sents(choices, sent[0])
            test_data.append(sent[1])
            data.append(test_data)

    return data

def make_data_from_all_words(data_pair, choices_lists, all_words):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        for choice in choices:
            if(len(choice.split(' '))>1):
                flag=-1
                #選択肢に2語以上のものがあるときはflagが負
        if(flag>0):
            test_data=make_sents(all_words, sent[0])
            test_data.append(sent[1])
            data.append(test_data)

    return data


def output_preds(file_name, preds):
    with open(file_name, 'w') as f:
        for p in preds:
            f.write(p+'\n')


def print_score(line, OK):
    print('  acc: ', '{0:.2f}'.format(1.0*OK/line*100),' %')
    print(' line: ',line)
    print('   OK: ',OK)


def output_score(file_name, line, allOK, clozeOK, partOK, BLEU, miss):
    output=''
    output=output+'  acc(all): '+str(1.0*allOK/line*100)+' %\n'
    output=output+'acc(cloze): '+str(1.0*clozeOK/line*100)+' %\n'
    output=output+' acc(part): '+str(1.0*partOK/line*100)+' %\n\n'
    output=output+'      BLEU: '+str(BLEU*100.0)+' %\n\n'
    output=output+'       all: '+str(allOK)+'\n'
    output=output+'     cloze: '+str(clozeOK)+'\n'
    output=output+'      part: '+str(partOK)+'\n'
    output=output+'      line: '+str(line)+'\n'
    output=output+'      miss: '+str(miss)+'\n'

    with open(file_name, 'w') as f:
        f.write(output)


def get_best_sent(sent_list, model):
    scores=[]
    for sent in sent_list:
        #score=model.score(sent,bos=True,eos=True)
        score=model.score(sent,bos=False,eos=False)
        score=score/len(sent.split(' '))
        scores.append(score)

    return sent_list[scores.index(max(scores))]



def calc_acc(data, model):
    line=0
    OK=0
    for one_data in data:
        line+=1
        ans=one_data[-1]
        ans=ans.replace('{ ', '')
        ans=ans.replace(' }', '')
        ans.strip()
        pred=get_best_sent(one_data[:len(one_data)-1], model)
        if pred == ans:
            OK+=1
    print_score(line, OK)



#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()

    # 1.語彙データ読み込み
    vocab_path=file_path+'enwiki_vocab30000_wordonly.txt'
    all_words = get_words(vocab_path)

    center_cloze=git_data_path+'center_cloze.txt'
    center_ans=git_data_path+'center_ans.txt'
    center_choi=git_data_path+'center_choices.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'
    MS_choi=git_data_path+'microsoft_choices.txt'

    print("Reading Testdata...")
    center_data=readData(center_cloze, center_ans)
    center_choices=get_choices(center_choi)

    MS_data=readData(MS_cloze, MS_ans)
    MS_choices=get_choices(MS_choi)


    #kenLMモデル読み込み
    model_N5=file_path+'text8_UNK_N5.klm'
    model_N4=file_path+'text8_UNK_N4.klm'
    model_N3=file_path+'text8_UNK_N3.klm'
    models=[model_N5, model_N4, model_N3]

    for model_path in models:
        print(model_path)
        print(datetime.datetime.today())
        model = kenlm.LanguageModel(model_path)
        print('{0}-gram model'.format(model.order))

        #テストデータに対する予測と精度の計算
        #選択肢を使ったテスト
        print('Use choices')
        print('center')
        data=make_data(center_data, center_choices, one_word=False)
        calc_acc(data, model)

        data=make_data(center_data, center_choices, one_word=True)
        calc_acc(data, model)

        print('MS')
        data=make_data(MS_data, MS_choices, one_word=False)
        calc_acc(data, model)

        data=make_data(MS_data, MS_choices, one_word=True)
        calc_acc(data, model)


        #選択肢なしテスト
        #空所内の選択肢は全て1語のみ
        print('\nNot use choices, from all words')
        print('center')
        data=make_data_from_all_words(center_data, center_choices, all_words)
        calc_acc(data, model)

        print('MS')
        data=make_data_from_all_words(MS_data, MS_choices, all_words)
        calc_acc(data, model)
