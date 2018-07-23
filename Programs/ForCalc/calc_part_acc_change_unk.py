# -*- coding: utf-8 -*-

'''
seq2seqの結果からBLEUとかaccとか計算
'''

from __future__ import print_function

import collections
import math
import codecs
import os
import re
import subprocess

import tensorflow as tf



#----- グローバル変数一覧 -----




#----- 自作関数 -----


def change_unk(line, vocab_set):
    li=line.split(' ')
    len_li=len(li)
    flag=0
    #↓ここもっとスマートな書き方ありそう
    for i in range(len_li):
        if not li[i] in vocab_set:
            flag=1
            li[i]='<unk>'
    out=' '.join(li)
    if flag>0:
        print(line)
        print(out)
    return out




def make_vocab_set(vacab_path):
    vocab_set=set()
    with open(vacab_path) as f:
        for line in f:
            line=re.sub(r'[^a-z{}<> ]', '', line)
            vocab_set.add(line)

    return vocab_set



def is_correct_cloze(line):
    left=line.count('{')
    right=line.count('}')
    if left*right==1:
        return True
    elif left+right>1:
        #print(line)
        pass
    else:
        #print(line)
        pass
    return False



def check_cloze(ref, ans):
    ref=re.sub(r'.*{', '', ref)
    ref=re.sub(r'}.*', '', ref)

    ans=re.sub(r'.*{', '', ans)
    ans=re.sub(r'}.*', '', ans)

    return ref==ans


def check_sent(ref, ans):
    ref=re.sub(r'{.*}', '', ref)
    ans=re.sub(r'{.*}', '', ans)

    return ref==ans

def get_cloze(ref, ans):
    ref=re.sub(r'.*{ ', '', ref)
    ref_cloze=re.sub(r' }.*', '', ref)

    ans=re.sub(r'.*{ ', '', ans)
    ans_cloze=re.sub(r' }.*', '', ans)

    return ref_cloze, ans_cloze

def get_long_length(ref_cloze, ans_cloze):
    ref_li=ref_cloze.split(' ')
    ans_li=ans_cloze.split(' ')
    ref_len=len(ref_li)
    ans_len=len(ans_li)
    if ref_len < ans_len:
        return ans_len

    return ref_len


def get_ans_length(ans_cloze):
    ans_li=ans_cloze.split(' ')
    ans_len=len(ans_li)

    return ans_len

def match(ref_cloze, ans_cloze):
    ref_set=set(ref_cloze.split(' '))
    ans_set=set(ans_cloze.split(' '))
    i=0
    '''
    print(ref_set)
    print(ans_set)
    print()
    '''
    for word in ref_set:
        if word in ans_set:
            i+=1

    return i





def calc_part_acc(ref_path, ans_path, vocab_set):
    match_sum=0
    leng_sum=0
    ct=0
    line_num=0
    match_line_num=0
    with open(ref_path) as f_ref:
        with open(ans_path) as f_ans:
            for ref_line in f_ref:
                ans_line=f_ans.readline()
                ref_line=re.sub(r'[^a-z{}<> ]', '', ref_line)
                ans_line=re.sub(r'[^a-z{}<> ]', '', ans_line)
                ref_line=change_unk(ref_line, vocab_set)
                ans_line=change_unk(ans_line, vocab_set)
                ref_cloze,ans_cloze=get_cloze(ref_line, ans_line)
                tmp_ans_length=get_ans_length(ans_cloze)
                leng_sum+=tmp_ans_length
                line_num+=1
                if is_correct_cloze(ref_line):
                    tmp_match=match(ref_cloze, ans_cloze)
                    match_sum+=tmp_match
                    if tmp_match > 0:
                        match_line_num+=1
                        #print(ref_line)
                        #print(ans_line)

                    if tmp_ans_length == tmp_match:
                        #print(ref_line)
                        #print(ans_line)
                        ct+=1

    print(match_sum)
    print(leng_sum)
    print(line_num)
    print(match_line_num)

    print(ct)
    return 100.0*match_sum/leng_sum, 100.0*match_line_num/line_num



def print_result(ref_path, ans_path, vocab_set):
    print('file: ',ref_path)
    acc1, acc2=calc_part_acc(ref_path, ans_path, vocab_set)
    print('part_acc_word: ','{0:.2f}'.format(acc1))
    print('part_acc_line: ','{0:.2f}'.format(acc2))

#return なし


#----- いわゆるmain部みたいなの -----
vocab_set=make_vocab_set('/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/enwiki_vocab30000.cloze')

print_result('output_dev.txt','/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt_dev.ans', vocab_set)
print('\n\n')

print_result('output_infer.txt','/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/center_nmt.ans', vocab_set)
