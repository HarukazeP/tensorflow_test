# -*- coding: utf-8 -*-

'''
linuxのsplitコマンドで分割したwikiファイルを読み取り、
kenlm用に20〜40語に分割されたファイルを作成

いきなり24.9GBとかのファイルを読み込むのはメモリエラー？でフリーズしてしまう
5GBでもフリーズしたから
split -b 1GB -d enwiki20171202.txt wiki_splited_
で1GBずつ分割してる
'''

from __future__ import print_function

import numpy as np
import datetime
import os
import os.path
import glob


#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today



#listの各要素を単語で連結してstring型で返す
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#文の長さの乱数
def rand_sent():
    li=list(range(1,41))
    we=[ 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.03135889, 0.06271777, 0.06271777, 0.09756098, 0.1358885, 0.09407666, 0.08710801, 0.10801394, 0.08362369, 0.04181185, 0.05574913, 0.04529617, 0.05574913, 0.03832753, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    we[15]=1.0-sum(we)+we[15]


    return np.random.choice(li, p=we)


#空所の長さの乱数
def rand_cloze():
    li=list(range(1,6))
    we=[0.615502686109, 0.235610130468, 0.114351496546, 0.0283960092095, 0.00613967766692]
    we[0]=1.0-sum(we)+we[0]
    return np.random.choice(li, p=we)



#学習データへの前処理を行う
#小文字化，アルファベット以外の文字の削除，1万単語ごとに分割

def parse_line_faster_for_filelist(input_file_list, new_path):
    if (os.path.exists(new_path)==False):
        print('Preprpcessing training data...')
        text=''
        with open(new_path, 'w') as f_out:
            for input_path in input_file_list:
                with open(input_path) as f_in:
                    print_time(input_path)
                    for line in f_in:
                        i=0
                        line=line.strip()
                        line_list=line.split(' ')
                        del line
                        #最後は変になってると思うので削除
                        line_len=len(line_list)-1
                        line_list=line_list[:line_len]
                        #print_time('Write start')
                        while(i<line_len):
                            max_len=rand_sent()
                            try:
                                text=list_to_sent(line_list,i,i+max_len)
                                i+=max_len
                                f_out.write(text+'\n')
                                #text8形式のコーパスだけを想定してる
                            except KeyError:
                                break
                        del line_list

        print_time('preprpcess end')


#----- いわゆるmain部みたいなの -----

#開始時刻のプリント
start_time=print_time('all start')

#データ
#tmp_path='text8.txt'
file_path='/media/tamaki/HDCL-UT/tamaki/M2/data_for_kenlm/wiki_split/wiki_splited_*'
file_list=sorted(glob.glob(file_path))
output_path='/media/tamaki/HDCL-UT/tamaki/M2/data_for_kenlm/enwikiALL'
tmp_path=output_path+'_splited.txt'

parse_line_faster_for_filelist(file_list, tmp_path)
