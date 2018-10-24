# -*- coding: utf-8 -*-

'''
kenLMライブラリを用いた統計言語モデル（n-grams）
https://github.com/kpu/kenlm

#TODO テスト関連の再確認


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
def make_data_one_word(data_pair, choices_lists):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        for choice in choices:
            if(len(choice.split(' '))>1):
                flag=-1
        if(flag>0):
            test_data=make_sents(choices, sent[0])
            test_data.append(sent[1])
            data.append(test_data)

    return data


#精度いろいろ計算
#問題文、完全一致文、空所の完答文、空所の一部正答文、BLEU値、空所ミス文
def calc_score(preds_sentences, ans_sentences):
    line_num=0
    allOK=0
    clozeOK=0
    partOK=0
    miss=0

    for pred, ans in zip(preds_sentences, ans_sentences):
        pred=pred.replace(' <EOS>', '')
        flag=0
        if pred == ans:
            allOK+=1
            flag=1
        pred_cloze = get_cloze(pred)
        ans_cloze = get_cloze(ans)
        tmp_ans_length=len(ans_cloze.split(' '))
        line_num+=1
        if is_correct_cloze(pred):
            tmp_match=match(pred_cloze, ans_cloze)
            if tmp_match > 0:
                partOK+=1
            if pred_cloze == ans_cloze:
                clozeOK+=1
                if flag==0:
                    print(pred)
                    print(ans)
        else:
            miss+=1

    BLEU=compute_bleu(preds_sentences, ans_sentences)

    return line_num, allOK, clozeOK, partOK, BLEU, miss


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
    max=-10000000
    best_sent=''
    for sent in sent_list:
        score=model.score(sent)
        #print(score)
        if(max<score):
            max=score
            best_sent=sent
    return best_sent




def calc_acc(data, model):
    line=0
    OK=0
    for one_data in data:
        line+=1
        pred=get_best_sent(one_data[:len(one_data)-1], model)
        print(pred)
        print(ans)
        if pred == one_data[-1]:
            OK+=1
    print_score(line, OK)



#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=100)
    #TODO ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()

    # 1.語彙データ読み込み
    #vocab_path=file_path+'enwiki_vocab30000.txt'
    #vocab = readVocab(vocab_path)

    test_cloze=file_path+'center_cloze.txt'
    test_ans=file_path+'center_ans.txt'
    test_choi=file_path+'center_choices.txt'

    print("Reading data...")
    test_data=readData(test_cloze, test_ans)
    choices=get_choices(test_choi)

    #kenLMモデル読み込み
    model_path=file_path+'text8.klm'
    model = kenlm.LanguageModel(model_path)
    print('{0}-gram model'.format(model.order))

    #テストデータに対する予測と精度の計算
    #選択肢を使ったテスト
    data=make_data_one_word(test_data, choices)
    calc_acc(data, model)
    '''
    print(len(data))

    for i in range(10):
        print(data[i][0])
    '''
