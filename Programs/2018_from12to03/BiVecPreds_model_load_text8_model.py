# -*- coding: utf-8 -*-

'''
M1のときにやってた学習済みモデルをロードして、テスト
学習はしない

text8の前処理

当時はやってなかった補充文スコアでのテストもやる
テストデータも変わってる
結構かなり書き換え

#TODO まだ未作成

動かしていたバージョン
python  : 3.5.2


min_modelをロードしてテストする用

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

'''

from __future__ import print_function
from keras.models import Sequential, Model
from keras.models import model_from_json
from keras.layers import Dense, Activation, Input, Embedding
from keras.layers import LSTM, Bidirectional
from keras.layers import add
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import keras
import numpy as np
import re
import sys
import datetime
import os
import os.path
import subprocess

#----- グローバル変数一覧 -----
my_epoch=100
vec_size=100
KeyError_set=set()
save_path=''
tmp_vec_dict=dict()

CLZ_word= 'XXXX'

#----- 関数群 -----

#時間表示
def print_time(str1):
    today=datetime.datetime.today()
    print(str1)
    print(today)
    return today


#学習データやテストデータへの前処理
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
    after_line = re.sub(r'[^a-z{}]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)

    return after_line


#学習データやテストデータへの前処理
def preprocess_line_for_choices(before_line):
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
    after_line = re.sub(r'[^a-z{}#]', ' ', after_line)
    after_line = re.sub(r'[ ]+', ' ', after_line)

    return after_line


#listの各要素を単語で連結してstring型で返す
def list_to_sent(list_line, start, end):
    sent=' '.join(list_line[start:end])
    return sent


#fasttextのベクトルファイルから単語辞書とベクトル辞書の作成
def vec_to_dict(vec_path):
    print('Loading fasttext vec ...')
    s=set()
    word_indices=dict()
    indices_word=dict()
    vec_dict=dict()
    i=0
    text=''
    with open(vec_path,'r') as f:
        for line in f:
            if i!=0:
                #先頭行には単語数と次元数が書かれているので無視
                line=line.replace('\n', '').replace('\r','')
                if line[-1]==' ':
                    line=line[:-1]
                tmp_list=line.split(' ')
                word=tmp_list[0]
                str_list=tmp_list[1:]
                #辞書の作成
                #0番目はパディング用の数字なので使わないことに注意
                word_indices[word]=i
                indices_word[i]=word
                vec_dict[word]=np.array(str_list, dtype=np.float32)
            i+=1

    word_indices['#OTHER']=i
    indices_word[i]='#OTHER'
    len_words=i
    return len_words, word_indices, indices_word, vec_dict


#fasttextのベクトルを得る
#未知語の場合にはfasttextのモデル呼び出して実行
#未知語は集合に格納し，あとでファイル出力
def get_ft_vec(word, vec_dict, ft_path, bin_path):
    if word in vec_dict:
        return vec_dict[word]
    elif word in tmp_vec_dict:
        return tmp_vec_dict[word]
    else:
        KeyError_set.add(word)    #要素を追加
        cmd='echo "'+word+'" | '+ft_path+' print-word-vectors '+bin_path
        ret  =  subprocess.check_output(cmd, shell=True)
        line=ret.replace('\n', '').replace('\r','')
        if line[0]==' ':
            line=line[1:]
        if line[-1]==' ':
            line=line[:-1]
        tmp_list=line.split(' ')
        word=tmp_list[0]
        vec=tmp_list[1:]
        vec_array=np.array(vec,dtype=np.float32)
        tmp_vec_dict[word]=vec_array

        return vec_array


#空所つき英文読み取り
#空所はCLZ_wordに置換
def readCloze2(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=preprocess_line(line)
            line=re.sub(r'{.*}', CLZ_word, line)
            line = re.sub(r'[ ]+', ' ', line)
            data.append(line.strip())

    return data


#選択肢読み取り
def readChoices(file_name):
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=preprocess_line_for_choices(line)
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            line=line.strip()
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


#空所内の単語読み取り
def readAns(file_name):
    data=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=preprocess_line(line)
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            data.append(line.strip())

    return data


#2つのベクトルのコサイン類似度を返す
def calc_similarity(pred_vec, ans_vec):
    len_p=np.linalg.norm(pred_vec)
    len_a=np.linalg.norm(ans_vec)
    if len_p==0 or len_a==0:
        return 0.0
    return np.dot(pred_vec/len_p, ans_vec/len_a)



#単語から辞書IDを返す
def search_word_indices(word, word_to_id):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['#OTHER']


class ModelTest():
    def __init__(self, model, maxlen_words, word_to_id, vec_dict, ft_path, bin_path, id_to_word):
        self.model=model
        self.N=maxlen_words
        self.word_to_id=word_to_id
        self.vec_dict=vec_dict
        self.ft_path=ft_path
        self.bin_path=bin_path
        self.id_to_word=id_to_word

    #選択肢が全て1語かどうかのチェック
    def is_one_word(self, choices):
        for c in choices:
            if c.count(' ')>0:
                return False

        return True

    #直近予測スコアの算出
    #モデルの出力と各選択肢との類似度
    def calc_near_scores(self, cloze_sent, choices):
        scores=[]
        cloze_list=cloze_sent.split()
        clz_index=cloze_list.index(CLZ_word)

        f_X=cloze_list[clz_index-self.N:clz_index-1]
        r_X=cloze_list[clz_index+1:clz_index+self.N]
        #TODO　padding
        #TODO indexにもしてない
        #TODO r_Xの方は逆順にする？

        preds_vec = self.model.predict([f_X, r_X], verbose=0)

        #choices は必ず1語
        for word in choices:
            word_vec=get_ft_vec(word, self.vec_dict, self.ft_path, self.bin_path)
            score=calc_similarity(preds_vec, word_vec)
            scores.append(score)

        return scores


    def make_sent_list_for_sent_score(self, cloze_sent, choice_words):
        #paddingもやる

        #choices_wordsは1語だけとは限らない


        pass

        return sent_list

    #補充文スコアの算出
    #モデルの出力と、補充文でのそこの単語との類似度の積？
    #式確認、logとって和とか？
    def calc_sent_scores(self, cloze_sent, choices):
        scores=[]

        for words in choices:
            score=0
            sent_list=self.make_sent_list_for_sent_score(cloze_sent, words)
            sent_len=len(sent_list)

            #Nとかの数rangeのとこも要確認
            for i in range(sent_len-2*self.N-1):
                #長さ計って、iとかでforループ？
                f_X=sent_list[i : i+self.N-1]
                r_X=sent_list[i+self.N+1 : i+2*self.N]
                tmp_word=sent_list[i+self.N]

                word_vec=get_ft_vec(tmp_word, self.vec_dict, self.ft_path, self.bin_path)
                #TODO ここの数の計算あってる？ Nとか +1とか
                preds_vec = self.model.predict([f_X, r_X], verbose=0)
                score=calc_similarity(preds_vec, word_vec)

                score+=score
                #scoreの対数化とか
                #長さで割るのも

            scores.append(score)


        return scores


    #直近予測スコア
    def check_one_sent_by_near_score(self, cloze_sent, choices, ans_word):
        line=0
        OK=0
        if self.is_one_word(choices):
            ans_index=choices.index(ans_word)
            line=1
            scores=self.calc_near_scores(cloze_sent, choices)
            if ans_index==scores.index(max(scores)):
                OK=1

        return line, OK


    #補充文スコア
    def check_one_sent_by_sent_score(self, cloze_sent, choices, ans_word, one_word=True):
        line=0
        OK=0
        ans_index=choices.index(ans_word)
        #1語のとき
        if self.is_one_word(choices):
            line=1
            scores=self.calc_sent_scores(cloze_sent, choices)
            if ans_index==scores.index(max(scores)):
                OK=1

        #1語以上もテストするとき
        elif one_word==False:
            line=1
            scores=self.calc_sent_scores(cloze_sent, choices)
            if ans_index==scores.index(max(scores)):
                OK=1

        return line, OK


    #テスト
    def model_test_both_score(self, data_name, cloze_path, choices_path, ans_path):
        '''
        ファイル読み込み
        モデル入力作成、1単語限定かどうかとか
        スコア計算とか

        選択肢のみ使用

        '''
        print(data_name)
        near_line=0
        near_OK=0

        sent_line_one_word=0
        sent_OK_one_word=0

        sent_line=0
        sent_OK=0

        #ファイル読み込み、どのスコアでも共通

        #空所はCLZ_wordに置換したやつ
        cloze_list=readCloze2(cloze_path)
        choices_list=readChoices(choices_path)
        ans_list=readAns(ans_path)

        for cloze_sent, choices, ans_word in zip(cloze_list, choices_list, ans_list)
            #直近予測スコア(1語のみ)
            line, OK=check_one_sent_by_near_score(cloze_sent, choices, ans_word)
            near_line+=line
            near_OK+=OK

            #補充文スコア(1語のみ)
            line, OK=check_one_sent_by_sent_score(cloze_sent, choices, ans_word, one_word=True)
            sent_line_one_word+=line
            sent_OK_one_word+=OK

            #補充文スコア(1語以上)
            line, OK=check_one_sent_by_sent_score(cloze_sent, choices, ans_word, one_word=False)
            sent_line+=line
            sent_OK+=OK

        print('near score')
        print('line:%d, acc:%.4f'% (near_line, 1.0*near_OK/near_line))

        print('sent score (one word)')
        print('line:%d, acc:%.4f'% (sent_line_one_word, 1.0*sent_OK_one_word/sent_line_one_word))

        print('sent score')
        print('line:%d, acc:%.4f'% (sent_line, 1.0*sent_OK/sent_line))


        return result_str



#----- main部 -----
if __name__ == '__main__':
    # 0.いろいろ前準備
    #開始時刻のプリント
    start_time=print_time('all start')
    start_time_str = start_time.strftime('%Y_%m_%d_%H%M')

    #モデルとか結果とかを格納するディレクトリの作成
    argvs = sys.argv
    argc = len(argvs)
    if argc <3:    #ファイル名 min_modelのパス wの長さで3つ必要(python は含まれない)
        print('### ERROR: invalid argument! ###')
    min_model_path=argvs[1] #ここ実行時の第二引数、〜〜.jsonのファイル
    #例： 2018_01_04_1450epoch100_e100_w10_add_bilstm_den1/min_model/
    maxlen_words=int(argvs[2])
    save_path=min_model_path[:min_model_path.find('min_model')]   #save_pathはmin_modelの手前の/まで
    save_path=save_path+'NEW_TEST_'


    #学習データの候補
    train_big='../corpus/WikiSentWithEndMark1.txt'   # 約5.8GB，約2000万行
    train_enwiki='../corpus/enwiki.txt'   # 約24GB，1行のみ，約435億単語(約237種類)
    train_mid='../corpus/miniWiki_tmp8.txt'   # 約1.5MB，約5000行
    train_small='../corpus/nietzsche.txt'   # 約600KB，約1万行
    train_test='../corpus/mini_text8.txt'

    train_text8='../corpus/text8.txt'   # 約95MB 1行のみ, 約1700万単語(約7万これ違う種類)  http://mattmahoney.net/dc/text8.zip


    # 2.fasttextのロードと辞書の作成
    '''
    https://github.com/facebookresearch/fastText
    このfastextを事前に実行しておき，その結果を利用
    '''
    ft_path='../../FastText/fastText-0.1.0/fasttext'

    #ベクトルファイルの候補
    vec_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.vec'
    bin_enwiki='../../FastText/Model/enwiki_dim'+str(vec_size)+'_minC0.bin'
    vec_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.vec'
    bin_text8='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.bin'

    #実際に使うもの
    vec_path=vec_text8
    bin_path=bin_text8

    len_words, word_to_id, id_to_word, vec_dict=vec_to_dict(vec_path)

    #embeddingで用いる，単語から行列への変換行列
    embedding_matrix = np.zeros((len_words+1, vec_size))
    for i in range(len_words):
        if i!=0:
            #IDが0の単語が存在しないので0は飛ばす
            embedding_matrix[i] = get_ft_vec(id_to_word[i], vec_dict, ft_path, bin_path)

    end_data=print_time('prepare data and fasttext end')



    # 5.val_loss最小モデルのロード
    min_model_file=min_model_path+'my_model.json'
    min_weight_file=min_model_path+'my_model.h5'
    print('Loading  '+min_model_file)

    json_string = open(min_model_file).read()
    min_model = model_from_json(json_string)
    min_model.load_weights(min_weight_file)
    optimizer = RMSprop()
    min_model.compile(loss='mean_squared_error', optimizer=optimizer)

    end_load=print_time('Load min_model end')


    # 6.テストの実行
    center_cloze=git_data_path+'center_cloze.txt'
    center_choi=git_data_path+'center_choices.txt'
    center_ans=git_data_path+'center_ans.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_choi=git_data_path+'microsoft_choices_for_CLOTH.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'

    high_path=git_data_path+'CLOTH_test_high'
    middle_path=git_data_path+'CLOTH_test_middle'

    CLOTH_high_cloze = high_path+'_cloze.txt'
    CLOTH_high_choi = high_path+'_choices.txt'
    CLOTH_high_ans = high_path+'_ans.txt'

    CLOTH_middle_cloze = middle_path+'_cloze.txt'
    CLOTH_middle_choi = middle_path+'_choices.txt'
    CLOTH_middle_ans = middle_path+'_ans.txt'


    center_data=['center', center_cloze, center_choi, center_ans]
    MS_data=['MS', MS_cloze, MS_choi, MS_ans]
    CLOTH_high_data=['CLOTH_high', CLOTH_high_cloze, CLOTH_high_choi, CLOTH_high_ans]
    CLOTH_middle_data=['CLOTH_middle', CLOTH_middle_cloze, CLOTH_middle_choi, CLOTH_middle_ans]

    datas=[center_data, MS_data, CLOTH_high_data, CLOTH_middle_ans]

    test=ModelTest(min_model, maxlen_words, word_to_id, vec_dict, ft_path, bin_path, id_to_word)

    for data in datas:
        data_name=data[0]
        cloze_path=data[1]
        choices_path=data[2]
        ans_path=data[3]

        test.model_test_both_score(data_name, cloze_path, choices_path, ans_path)

    end_test=print_time('test end')
