# -*- coding: utf-8 -*-

'''
seq2seq�p��text8�R�[�p�X����w�K�f�[�^�ƌ��؃f�[�^���쐬

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
def preprocess_line(before_line):
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
    after_line = re.sub(r'[^(a-z|\{|\})]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)

    return after_line


#list�̊e�v�f��P��ŘA������string�^�ŕԂ�
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#���̒����̗���
def rand_sent():
    li=list(range(1,41))
    we=[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03135889, 0.06271777, 0.06271777, 0.09756098, 0.1358885, 0.09407666, 0.08710801, 0.10801394, 0.08362369, 0.04181185, 0.05574913, 0.04529617, 0.05574913, 0.03832753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    we[15]=1.0-sum(we)+we[15]
    

    return np.random.choice(li, p=we)


#�󏊂̒����̗���
def rand_cloze():
    li=list(range(1,6))
    we=[0.615502686109, 0.235610130468, 0.114351496546, 0.0283960092095, 0.00613967766692]
    we[0]=1.0-sum(we)+we[0]
    return np.random.choice(li, p=we)

#�w�K�f�[�^�ւ̑O�������s��
#���������C�A���t�@�x�b�g�ȊO�̕����̍폜�C1���P�ꂲ�Ƃɕ���
def parse_line(old_path, new_path):
    if (os.path.exists(new_path)==False):
        print('Preprpcessing training data...')
        text=''
        text_len=0
        i=0
        with open(old_path) as f_in:
            with open(new_path, 'w') as f_out:
                for line in f_in:
                    #���̑O������text8�Ƃ��̑O�����Ɠ���
                    line=preprocess_line(line)
                    line_list=line.split(' ')
                    line_len=len(line_list)
                    #max_len�ȉ��̎��͘A�����Ď���
                    max_len=rand_sent()
                    if(text_len+line_len <= max_len):
                        if(text_len==0):
                            text=line
                        else:
                            text=text+' '+line
                        text_len=text_len+line_len
                    #max_len��蒷���Ƃ���max_len�P�ꂲ�Ƃɋ�؂��ăt�@�C���֏�������
                    else:
                        while (line_len>max_len):
                            if(text_len==0):
                                text=list_to_sent(line_list,0,max_len)
                            else:
                                text=text+' '+list_to_sent(line_list,0,max_len-text_len)
                            f_out.write(text+'\n')
                            text=''
                            text_len=0
                            #�c��̍X�V
                            line_list=line_list[max_len-text_len+1:]
                            line_len=len(line_list)
                            max_len=rand_sent()
                        #while �I���i1�s�̖����̏����j
                        #�]��͎��̍s�ƘA��
                        text=list_to_sent(line_list,0,line_len)
                        text_len=line_len
                #for�I���i�t�@�C���̍Ō�̍s�̏����j
                if text_len!=0:
                    text=preprocess_line(text)
                    f_out.write(text+'\n')
                print('total '+str(i)+' line\n')
                print_time('preprpcess end')

    return new_path




def make_data(old_path, cloze_path, ans_path):
    if (os.path.exists(ans_path)==False):
        print('make data start...')
        i=0
        with open(old_path) as f_in:
            with open(cloze_path, 'w') as f_cloze:
                with open(ans_path, 'w') as f_ans:
                    for line in f_in:
                        i+=1
                        print('line: '+str(i)+'\n')
                        line_list=line.split(' ')
                        line_len=len(line_list)
                        if line_len > 8:
                            #cloze_len��1�`5�̂ǂꂩ
                            cloze_len=rand_cloze()
                            #cloze_start��0�`
                            #�Ⴆ��1�s��10�P�ꂠ���āC�󏊂�2�P��Ȃ�C
                            #randint(9)�ƂȂ�Ccloze_start��0�`8
                            cloze_start=np.random.randint(line_len-cloze_len+1)
                            cloze_end=cloze_start+cloze_len+1
                            
                            line_list.insert(cloze_start, '{')
                            line_list.insert(cloze_end, '}')
                            ans_text=' '.join(line_list)
                            cloze_text=re.sub('\{.+\}', '{ }',ans_text)
                            
                            f_ans.write(ans_text+'\n')
                            f_cloze.write(cloze_text+'\n')

        print_time('make data end')




#----- ������main���݂����Ȃ� -----

#�J�n�����̃v�����g
start_time=print_time('all start')


#�f�[�^
tmp_path='../../Data/my_nmt/tmp_wiki.txt'   # ��95MB 1�s�̂�, ��1700���P��(��7�����)  http://mattmahoney.net/dc/text8.zip

print('Loading  '+tmp_path)
file_name=tmp_path[:-4]
tmp2_path=parse_line(tmp_path, file_name+'_nmt_tmp.txt')
make_data(tmp2_path, file_name+'_nmt_cloze.txt', file_name+'_nmt_ans.txt')



