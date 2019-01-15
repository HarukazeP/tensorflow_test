# -*- coding: utf-8 -*-

'''
CLOTHデータセット(json)から
今までのデータと同じtxt形式で作成

文ごとに分割とか

'''

from __future__ import print_function

import re
import datetime

import json
from collections import OrderedDict
import nltk
import glob
import unicodedata


#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today

def ans_id_to_word(choices_list, ans_id):
    if ans_id=='A':
        ans_word=choices_list[0]
    elif ans_id=='B':
        ans_word=choices_list[1]
    elif ans_id=='C':
        ans_word=choices_list[2]
    elif ans_id=='D':
        ans_word=choices_list[3]

    return ans_word


def make_test_sents(sent, choices_list, ans_id):
    sent = sent.replace('___', ' ___ ')
    sent = re.sub(r'[ ]+', ' ', sent)

    if sent.count(' ___ ')==0:
        print(sent)

    cloze=sent.replace('___', '{ }')
    choice_all=' ### '.join(choices_list)
    before=re.sub(r'{.*', '{ ', cloze)
    after=re.sub(r'.*}', ' }', cloze)
    choice = before + choice_all + after

    ans_word=ans_id_to_word(choices_list, ans_id)

    ans = before + ans_word + after

    return cloze, ans, choice

#1文に複数の空所を含むときは、複製し、空所の数が1つの文を複数にする
def make_new_sent(ct, tmp, sent, choi_id, sents, choices_list, ans_id, file_name):
    all_sent=sent.split('___')

    add_sent=[]
    for i in range(ct):
        out=''
        for j in range(ct):
            out+=all_sent[j]
            if i==j:
                out+=' ___ '
            else:
                out+=ans_id_to_word(choices_list[choi_id+j], ans_id[choi_id+j])
        out+=all_sent[ct]
        out = re.sub(r'[ ]+', ' ', out)
        add_sent.append(out)

    if tmp=='':
        for new_sent in add_sent:
            sents.append(new_sent)
    else:
        for new_sent in add_sent:
            out_sent=tmp+' '+new_sent
            if len(out)<80:
                sents.append(out_sent)
            else:
                sents.append(new_sent)
        tmp=''

    choi_id+=ct



def split_sent_list(sent_tokenize_list, choi_list, ans_list, file_name):
    tmp=''
    sents=[]
    flag=0
    choi_id=0

    for sent in sent_tokenize_list:
        ct=sent.count('___')
        #空所を含まない文は次に連結する
        if ct==0:
            if tmp=='':
                tmp=sent
            else:
                out=tmp+' '+sent
                if len(out)<40:
                    tmp=out
                else:
                    tmp=sent
        #空所が1つ(文の分割が正しいとき)
        elif ct==1:
            choi_id+=1
            if tmp=='':
                sents.append(sent)
            else:
                out=tmp+' '+sent
                if len(out)<80:
                    sents.append(out)
                else:
                    sents.append(sent)
                tmp=''
        # nltkの文の分割がおかしいときの処理(仮)
        else:
            #print('#ERROR1 : ',sent)
            new_sent = sent.replace('.', ' . ')
            new_sent = new_sent.replace('___', ' ___ ')
            new_sent = new_sent.replace('!', ' ! ')
            new_sent = new_sent.replace('?', ' ? ')
            new_sent = new_sent.replace(':', ' : ')
            new_sent = new_sent.replace(';', ' ; ')
            new_sent = re.sub(r'[ ]+', ' ', new_sent)
            for sent in nltk.sent_tokenize(new_sent):
                ct=sent.count('___')
                #空所を含まない文は次に連結する
                if ct==0:
                    if tmp=='':
                        tmp=sent
                    else:
                        out=tmp+' '+sent
                        if len(out)<40:
                            tmp=out
                        else:
                            tmp=sent
                #空所が1つ(文の分割が正しいとき)
                elif ct==1:
                    choi_id+=1
                    if tmp=='':
                        sents.append(sent)
                    else:
                        out=tmp+' '+sent
                        if len(out)<80:
                            sents.append(out)
                        else:
                            sents.append(sent)
                        tmp=''
                else:
                    #print('#ERROR2 : ',sent)
                    make_new_sent(ct, tmp, sent, choi_id, sents, choi_list, ans_list, file_name)
                    flag=1

    return sents, flag


#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )



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
    text=unicodeToAscii(text)
    text = text.replace('\"', ' ')
    text = text.replace('\n', ' ')
    text = text.replace('\r', ' ')

    text = text.replace('<IMG>', '')
    
    #ここ追加したやつ(2019/1/14)
    text = text.replace("[KS5UKS5U]", "")
    text = text.replace("[:Z|xx|k.Com]", "")
    text = text.replace("(;)", "")

    text = text.replace('{', ' ')
    text = text.replace('}', ' ')
    text = text.replace('<', ' ')
    text = text.replace('>', ' ')

    text = text.replace(' _', ' _ ')
    text = text.replace('_ ', ' _ ')
    text = text.replace(' _ ', ' ___ ')
    text = text.replace(' _ ', ' ___ ')
    text = re.sub(r'[ ]+', ' ', text)
    sent_tokenize_list = nltk.sent_tokenize(text)
    sents, flag=split_sent_list(sent_tokenize_list, choices_list, ans_list, json_path)
    '''
    for check_sent in sents:
        if len(check_sent.split())<4 and check_sent.count('___')==0:
            print(json_path)
            print(check_sent)

    if flag==1:
        print(json_path)

    '''
    sents_len=len(sents)

    if sents_len!=len(choices_list):
        print('sents', sents_len)
        print('text', text.count('___'))
        print('choices', len(choices_list))
        print('ans', len(ans_list))

        print(json_path)

        print(text)
        print('\n################################\n')

        for i in range(sents_len):
            print(sents[i])


        exit()

    cloze_file = output_pass+'_cloze.txt'
    ans_file = output_pass+'_ans.txt'
    choice_file = output_pass+'_choices.txt'

    before=''
    with open(cloze_file, 'a') as f_clz:
        with open(ans_file, 'a') as f_ans:
            with open(choice_file, 'a') as f_cho:
                for i in range(sents_len):
                    cloze, ans, choice=make_test_sents(sents[i], choices_list[i], ans_list[i])
                    if ans.count('<IMG>')>0:
                        print(json_path)
                    '''
                    if before=='':
                        before=ans
                    else:
                        if len(ans.split(' '))<6:
                            print(json_path)
                            for i in range(sents_len):
                                print(sents[i])
                            #exit()
                        else:
                            before=ans
                    '''


                    f_clz.write(cloze+'\n')
                    f_ans.write(ans+'\n')
                    f_cho.write(choice+'\n')

    return flag





def file_loop(in_head, out_head):

    high_path = '/high/high*.json'
    middle_path = '/middle/middle*.json'

    #ここ書いた後順番入れ替えたからコード汚い
    print_time('make train data')
    file2 = sorted(glob.glob(in_head + 'train' + middle_path))
    for file_name in file2:
        json_to_text(file_name, out_head + 'train')

    file1 = sorted(glob.glob(in_head + 'train' + high_path))
    for file_name in file1:
        json_to_text(file_name, out_head + 'train')

    print_time('make valid data')
    file4 = sorted(glob.glob(in_head + 'valid' + middle_path))
    for file_name in file4:
        json_to_text(file_name, out_head + 'valid')

    file3 = sorted(glob.glob(in_head + 'valid' + high_path))
    for file_name in file3:
        json_to_text(file_name, out_head + 'valid')

    print_time('make test data')
    file6 = sorted(glob.glob(in_head + 'test' + middle_path))
    for file_name in file6:
        json_to_text(file_name, out_head + 'test_middle')

    file5 = sorted(glob.glob(in_head + 'test' + high_path))
    for file_name in file5:
        json_to_text(file_name, out_head + 'test_high')






#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
tmp_path='../../../pytorch_data/'
input_path_head=tmp_path+'myCLOTH/'
output_path_head=tmp_path+'CLOTH_for_model/CLOTH_'

file_loop(input_path_head, output_path_head)
