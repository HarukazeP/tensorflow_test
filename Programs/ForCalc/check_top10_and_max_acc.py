# -*- coding: utf-8 -*-

'''
PyTorchのモデル関連
正答で空所に入る語の分布確認と
与えた語彙に正答が含まれているか(正答率の上限)
'''

from __future__ import print_function

import math
import os
import re
import unicodedata


def get_words(file):
    words=[]
    #print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=normalizeString(line)
            for word in line.split(' '):
                if not word in words:
                    words.append(word)

    return words


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

def in_vocab(words, vocab, all=True):
    ct=0
    for word in words:
        if word in vocab:
            ct+=1
    if all:
        if ct==len(words):
            return 1
    else:
        if ct>0:
            return 1

    return 0

def print_top10words_and_max_acc(file_path, vocab):
    words_d=dict()
    part_max_acc=0
    all_max_acc=0
    line=0
    with open(file_path) as f:
        for line in f:
            line+=1
            line=re.sub(r'[^a-z{}<> ]', '', line)
            line=normalizeString(line)
            line_cloze=re.sub(r'.*{ ', '', line)
            line_cloze=re.sub(r' }.*', '', line_cloze)
            line_cloze=line_cloze.strip()
            words=line_cloze.split(' ')
            part_max_acc+=in_vocab(words, vocab, all=False)
            all_max_acc+=in_vocab(words, vocab, all=True)
            for x in words:
                if x in words_d:
                    words_d[x]+=1
                else:
                    words_d[x]=1

    words_list = sorted(words_d.items(), key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(words_list[i])
    print(len(words_list))
    print('part max acc: ', 1.0*part_max_acc/line*100)
    print('all max acc : ', 1.0*all_max_acc/line*100)



def print_result(file_path, vocab_files):
    for vocab_path in vocab_files:
        print('sent file : ', file_path)
        print('vocab file: ', vocab_path)
        vocab=get_words(vocab_path)

        print_top10words_and_max_acc(file_path, vocab)




#return なし


#----- いわゆるmain部みたいなの -----

file_path='../../../pytorch_data/'
git_data_path='../../Data/'

center_ans=git_data_path+'center_ans.txt'
MS_ans=git_data_path+'microsoft_ans.txt'

vocab0=file_path+'enwiki_vocab30000.txt'
vocab1=file_path+'enwiki_vocab10000.txt'
vocab2=file_path+'enwiki_vocab1000.txt'
vocab3=file_path+'enwiki_vocab100.txt'

vocab_files=[vocab0, vocab1, vocab2, vocab3]



print_result(center_ans, vocab_files)
print_result(MS_ans, vocab_files)
