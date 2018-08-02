# -*- coding: utf-8 -*-

'''
空所外について、正誤判定
どれだけ空所の記号から離れたところか集計
'''

from __future__ import print_function

import collections
import math
import os
import re
import subprocess



#----- 自作関数 -----

def is_correct_cloze(line):
    left=line.count('{')
    right=line.count('}')
    if left*right==1:
        return True
    else:
        #print(line)
        pass
    return False


def check_before(pre, ans, OK, NG):
    pre_list=pre.split(' ')[::-1]
    ans_list=ans.split(' ')[::-1]
    for i in range(min(len(ans_list), len(pre_list))):
        if i<4:
            if pre_list[i]==ans_list[i]:
                OK[i]+=1
            else:
                NG[i]+=1
        else:
            if pre_list[i]==ans_list[i]:
                OK[4]+=1
            else:
                NG[4]+=1




def check_after(pre, ans, OK, NG):
    pre_list=pre.split(' ')
    ans_list=ans.split(' ')
    for i in range(min(len(ans_list), len(pre_list))):
        if i<4:
            if pre_list[i]==ans_list[i]:
                OK[i]+=1
            else:
                NG[i]+=1
        else:
            if pre_list[i]==ans_list[i]:
                OK[4]+=1
            else:
                NG[4]+=1



def check_distandce(predict_path, ans_path):
    #空所からの距離が1,2,3,4, と5以上
    OK_dis=[0, 0, 0, 0, 0]
    NG_dis=[0, 0, 0, 0, 0]
    with open(predict_path) as pre:
        with open(ans_path) as ans:
            for pre_line in pre:
                ans_line=ans.readline()
                pre_line=re.sub(r'[^a-z{}<> ]', '', pre_line)
                ans_line=re.sub(r'[^a-z{}<> ]', '', ans_line)
                if is_correct_cloze(pre_line):
                    if ans_line[0]=='{':
                        #空所が先頭
                        pre_after=pre_line.split(' } ')[-1]
                        ans_after=ans_line.split(' } ')[-1]
                        check_after(pre_after, ans_after, OK_dis, NG_dis)
                    elif ans_line[-1]=='}':
                        #空所が末尾
                        pre_before=pre_line.split(' { ')[0]
                        ans_before=ans_line.split(' { ')[0]
                        check_before(pre_before, ans_before, OK_dis, NG_dis)
                    else:
                        pre_before=pre_line.split(' { ')[0]
                        ans_before=ans_line.split(' { ')[0]
                        check_before(pre_before, ans_before, OK_dis, NG_dis)
                        pre_after=pre_line.split(' } ')[-1]
                        ans_after=ans_line.split(' } ')[-1]
                        check_after(pre_after, ans_after, OK_dis, NG_dis)
    OKsum=sum(OK_dis)
    newOK=[100.0*x/OKsum for x in OK_dis]
    NGsum=sum(NG_dis)
    newNG=[100.0*x/NGsum for x in NG_dis]

    print('OK(n): ', OK_dis)
    print('OK(%): ', newOK)
    print('NG(n): ', NG_dis)
    print('NG(%): ', newNG)






def print_result(predict_path, ans_path):
    print('\n\nfile: ',predict_path)
    check_distandce(predict_path, ans_path)
    #return なし


#----- いわゆるmain部みたいなの -----
dev_predict='/home/tamaki/M2/Tensorflow/nmt/text8_output_ep100_2/output_dev.txt'
dev_ans='/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt_dev.ans'

test_predict='/home/tamaki/M2/Tensorflow/nmt/text8_output_ep100_2/output_infer.txt'
test_ans='/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/center_nmt.ans'

print_result(dev_predict, dev_ans)
print_result(test_predict, test_ans)
