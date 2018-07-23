# -*- coding: utf-8 -*-

'''
�Z���^�[��text8�R�[�p�X�Ɠ��l�̑O����

'''

from __future__ import print_function

import numpy as np
import re
import sys
import datetime
import os
import os.path
import subprocess

#----- �O���[�o���ϐ��ꗗ -----
my_epoch=100
vec_size=100
maxlen_words = 5
KeyError_set=set()
today_str=''
tmp_vec_dict=dict()


#----- �֐��Q -----

#���ԕ\��
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today

#�w�K�f�[�^��e�X�g�f�[�^�ւ̑O����
def preprocess_line2(before_line):
    after_line=before_line.lower()
    after_line=after_line.replace('0', ' zero ')
    after_line=after_line.replace('1', ' one ')
    after_line=after_line.replace('2', ' two ')
    after_line=after_line.replace('3', ' three ')
    after_line=after_line.replace('4', ' four ')
    after_line=after_line.replace('5', ' five ')
    after_line=after_line.replace('6', ' six ')
    after_line=after_line.replace('7', ' seven ')
    after_line=after_line.replace('8', ' eight ')
    after_line=after_line.replace('9', ' nine ')
    after_line = re.sub(r'[^a-z{}\n]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)
    after_line = re.sub('^ ', '', after_line)
    after_line = re.sub(' \n', '\n', after_line)

    return after_line





#�w�K�f�[�^�ւ̑O�������s��
#���������C�A���t�@�x�b�g�ȊO�̕����̍폜�C1���P�ꂲ�Ƃɕ���
def clean_line(old_path, new_path):
    print('cleaning data...')
    with open(old_path) as f_in:
        with open(new_path, 'w') as f_out:
            for line in f_in:
                #���̑O������text8�Ƃ��̑O�����Ɠ���
                line=preprocess_line2(line)
                f_out.write(line)
    print('clean end')








#----- ������main���݂����Ȃ� -----

#�J�n�����̃v�����g
start_time=print_time('all start')


#�f�[�^
n100_path='../../Data/my_nmt/center_ans.txt'
n500_path='../../Data/my_nmt/center_cloze.txt'

clean_line(n100_path, '../../Data/my_nmt/center_nmt.ans')
clean_line(n500_path, '../../Data/my_nmt/center_nmt.cloze')


