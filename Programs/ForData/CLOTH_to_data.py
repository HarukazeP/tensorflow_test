# -*- coding: utf-8 -*-

'''
CLOTHデータセット(json)から
今までのデータと同じtxt形式で作成

文ごとに分割とか

'''

from __future__ import print_function

import re
import sys
import datetime
import os
import os.path

import json
from collections import OrderedDict
import nltk
import glob

#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today


def make_test_sents(sent, choices_list, ans_word):
    cloze=sent.replace('___', '{ }')
    choice_all=' ### '.join(choices_list)
    before=re.sub(r'{.*', '{ ', cloze)
    after=re.sub(r'.*}', ' }', cloze)
    choice = before + choice_all + after
    ans = before + ans_word + after

    return cloze, ans, choice



def split_sent_list(sent_tokenize_list):
    tmp=''
    sents=[]
    for sent in split_sent_list:
        ct=sent.count('___')
        #空所を含まない文は次に連結する
        if ct==0:
            if tmp=='':
                tmp=sent
            else:
                tmp=tmp+' '+sent
        #空所が1つ(文の分割が正しいとき)
        elif ct==1:
            if tmp=='':
                sents.add(sent)
            else:
                sents.add(tmp+' '+sent)
                tmp=''
        # nltkの文の分割がおかしいときの処理(仮)
        else:
            print('#ERROR1 : ',sent)
            new_sent = sent.replace('.', ' . ')
            new_sent = re.sub(r'[ ]+', ' ', new_sent)
            for sent in nltk.sent_tokenize(new_sent):
                ct=sent.count('___')
                #空所を含まない文は次に連結する
                if ct==0:
                    if tmp=='':
                        tmp=sent
                    else:
                        tmp=tmp+' '+sent
                #空所が1つ(文の分割が正しいとき)
                elif ct==1:
                    if tmp=='':
                        sents.add(sent)
                    else:
                        sents.add(tmp+' '+sent)
                        tmp=''
                else:
                    print('#ERROR2 : ',sent)
    return sents




def json_to_text(json_path, output_pass):
    with open(json_path) as f:
        #OrderedDict で順番を保持
        df = json.load(f, object_pairs_hook=OrderedDict)
    '''
    jsonファイル
        "article"   空所つき文章
                    文章のピリオドとｋはくっついてたり，離れてたりバラバラ
        "options"   選択肢
        "answers"   選択肢のid  A,B,C,D のどれか
        "source"    ファイル名

        ' _ ' ←空所はこれで単語扱いなので半角スペースが連続する

    '''

    text = df['article']   #文字列
    choices_list = df['options']   #配列
    ans_list = df['answers']   #配列

    #少しだけ前処理
    text = text.replace(' _ ', '___')
    text = re.sub(r'[ ]+', ' ', text)
    sent_tokenize_list = nltk.sent_tokenize(text)
    sents=split_sent_list(sent_tokenize_list)

    sents_len=len(sents)

    if sents_len!=len(choices_list):
        print('sents', sents_len)
        print('choices', len(choices_list))
        print('ans', len(ans_list))

        print(json_path)
        exit()

    cloze_file = output_pass+'_cloze.txt'
    ans_file = output_pass+'_ans.txt'
    choice_file = output_pass+'_choice.txt'

    with open(cloze_file, 'a') as f_clz:
        with open(ans_file, 'a') as f_ans:
            with open(choice_file, 'a') as f_cho:
                for i in range(len(sents_len)):
                    cloze, ans, choice=make_test_sents(sents[i], choices_list[i], ans_list[i])
                    f_clz.write(cloze+'\n')
                    f_ans.write(ans+'\n')
                    f_cho.write(choice+'\n')





def file_loop(in_head, out_head):

    high_path = '/high/high*.json'
    middle_path = '/middle/middle*.json'

    #ここ書いた後順番入れ替えたからコード汚い
    file2 = glob.glob(in_head + 'train' + middle_path)
    for file_name in file2:
        json_to_text(file_name, out_head + 'train')

    file1 = glob.glob(in_head + 'train' + high_path)
    for file_name in file1:
        json_to_text(file_name, out_head + 'train')

    file4 = glob.glob(in_head + 'valid' + middle_path)
    for file_name in file4:
        json_to_text(file_name, out_head + 'valid')

    file3 = glob.glob(in_head + 'valid' + high_path)
    for file_name in file3:
        json_to_text(file_name, out_head + 'valid')

    file6 = glob.glob(in_head + 'test' + middle_path)
    for file_name in file6:
        json_to_text(file_name, out_head + 'test_middle')

    file5 = glob.glob(in_head + 'test' + high_path)
    for file_name in file5:
        json_to_text(file_name, out_head + 'test_high')






#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
tmp_path='../../../'

input_path_head='hoge/CLOTH/'
output_path_head='hogehoge/CLOTH_'

file_loop(input_path_head, output_path_head)
