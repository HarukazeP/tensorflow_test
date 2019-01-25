# -*- coding: utf-8 -*-

'''

M1のときにやってた双方向ベクトル予測モデル
CLOTHデータセットで学習

CLOTH用の前処理

当時はやってなかった補充文スコアでのテストもやる
テストデータも変わってる

-------------------------------------------

動かしていたバージョン

python    : 2.7.12
keras     : 2.0.4
tensorflow: 1.1.0

'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import datetime
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse
import collections

import nltk

from keras import regularizers
from keras import backend as K
from keras.models import Model, model_from_json, load_model

from keras.layers import Dense, Activation, Input, Embedding
from keras.layers import LSTM
from keras.layers import add, concatenate, multiply
#from keras.utils.vis_utils import plot_model
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file

from glob import glob

import sys
import subprocess

#----- グローバル変数一覧 -----

vec_size=100
maxlen_words = 5
KeyError_set=set()
today_str=''
tmp_vec_dict=dict()
BATCH_SIZE=256

file_path='../../../pytorch_data/'
git_data_path='../../Data/'
CLOTH_path = file_path+'CLOTH_for_model/'

today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str


#事前処理いろいろ
print('Start: '+today_str)
CLZ_word='XXXX'

#----- 関数群 -----

###########################
# 1.データの準備，データ変換
###########################


#空所つき英文読み取り
#空所はCLZ_wordに置換
def readCloze2(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'{.*}', CLZ_word, line)
            line = re.sub(r'[ ]+', ' ', line)
            data.append(line.strip())

    return data


#選択肢読み取り
def readChoices(file_name):
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
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
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            data.append(line.strip())

    return data


#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


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
                line=line.strip()
                tmp_list=line.split(' ')
                word=tmp_list[0]
                str_list=tmp_list[1:]
                #辞書の作成
                #0番目はパディング用の数字なので使わないことに注意
                word_indices[word]=i
                indices_word[i]=word
                vec_dict[word]=np.array(str_list, dtype=np.float)
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
        #python3からここの出力がバイナリ列に変化
        line=ret.decode('utf-8').strip()
        tmp_list=line.split(' ')
        word=tmp_list[0]
        vec=tmp_list[1:]
        vec_array=np.array(vec,dtype=np.float32)
        tmp_vec_dict[word]=vec_array

        return vec_array


#単語から辞書IDを返す
def search_word_indices(word, word_to_id):
    if word in word_to_id:
        return word_to_id[word]
    else:
        return word_to_id['#OTHER']


###########################
# 2.モデル定義
###########################

# モデルの構築
def build_model(len_words, embedding_matrix):
    f_input=Input(shape=(maxlen_words,))
    f_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(f_input)

    f_full=Dense(vec_size,activation='relu')(f_emb)
    f_layer=LSTM(128)(f_full)

    r_input=Input(shape=(maxlen_words,))
    r_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(r_input)

    r_full=Dense(vec_size,activation='relu')(r_emb)
    r_layer=LSTM(128)(r_full)


    merged_layer=add([f_layer, r_layer])

    out_layer=Dense(vec_size,activation='relu')(merged_layer)


    my_model=Model([f_input, r_input], out_layer)

    optimizer = RMSprop()
    my_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return my_model

###########################
# 3.モデルの学習
###########################

def preprocess(s):
    sent_tokens=[]
    s = unicodeToAscii(s)
    s = re.sub(r'[ ]+', ' ', s)
    s = s.strip()
    tokens=nltk.word_tokenize(s)
    symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
    num_tag=("LS", "CD")
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if tag in symbol_tag:
            pass
            #記号は無視
        elif tag in num_tag:
            sent_tokens.append('NUM')
        else:
            sent_tokens.append(word.lower())

    return sent_tokens


#1行の文字列を学習データの形式に変換
def tokens_to_data(tokens, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path):
    f_X = []
    r_X = []
    Y = []
    len_text=len(tokens)
    leng=0

    ids=[]
    for word in tokens:
        ids.append(search_word_indices(word, word_to_id))

    len_text=len(ids)

    while(len_text < maxlen_words*2+1):
        ids=[0]+ids+[0]
        len_text+=2

    for i in range(len_text - maxlen_words*2 -1):
        f=ids[i: i + maxlen_words]
        r=ids[i + maxlen_words+1: i + maxlen_words+1+maxlen_words]
        n=ids[i + maxlen_words]
        f_X.append(f)
        r_X.append(r[::-1]) #逆順のリスト
        Y.append(get_ft_vec(id_to_word[n], vec_dict, ft_path, bin_path))

    return f_X, r_X, Y


#空所等を含まない英文のデータから，モデルの入出力を作成
def make_data(file_path, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path):
    f_X_list=[]
    r_X_list=[]
    Y_list=[]

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            tokens=preprocess(line)
            f, r, y=tokens_to_data(tokens, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path)
            f_X_list.extend(f)
            r_X_list.extend(r)
            Y_list.extend(y)
    f_X=np.array(f_X_list, dtype=np.int)
    r_X=np.array(r_X_list, dtype=np.int)
    Y=np.array(Y_list, dtype=np.float)

    return f_X, r_X, Y


#checkpoint で保存された最新のモデル(ベストモデルをロード)
def getNewestModel(model):
    files = [(f, os.path.getmtime(f)) for f in glob(save_path+'*hdf5')]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


#学習をn_iters回，残り時間の算出をlossグラフの描画も
def trainIters(model, train_path, val_path, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path, n_iters=5, print_every=10, saveModel=False):

    print('Make data for model...')
    f_X_train, r_X_train, Y_train=make_data(train_path, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path)
    f_X_val, r_X_val, Y_val=make_data(val_path, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path)


    X_train=[f_X_train, r_X_train]
    X_val=[f_X_val, r_X_val]

    cp_cb = ModelCheckpoint(filepath = save_path+'model_ep{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    start = time.time()
    st_time=datetime.datetime.today().strftime('%H:%M')
    print("Training... ", st_time)


    # Ctrl+c で強制終了してもそこまでのモデルで残りの処理継続
    try:
        hist=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=n_iters, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb], shuffle=True)

        #全学習終わり
        #lossとaccのグラフ描画
        showPlot3(hist.history['loss'], hist.history['val_loss'], 'loss.png', 'loss')

        print(hist.history['loss'])

    except KeyboardInterrupt:
        print()
        print('-' * 89)
        files = [(f, os.path.getmtime(f)) for f in glob(save_path+'*hdf5')]
        if len(files) > 0:
            print('Exiting from training early')
        else :
            exit()

    #ベストモデルのロード
    model=getNewestModel(model)

    return model


#グラフの描画（画像ファイル保存）
def showPlot3(train_plot, val_plot, file_name, label_name):
    fig = plt.figure()
    plt.plot(train_plot, color='blue', marker='o', label='train_'+label_name)
    plt.plot(val_plot, color='green', marker='o', label='val_'+label_name)
    plt.title('model '+label_name)
    plt.xlabel('epoch')
    plt.ylabel(label_name)
    plt.legend()
    plt.savefig(save_path + file_name)


#グラフの描画（画像ファイル保存）
def showPlot4(val_plot, file_name, label_name):
    fig = plt.figure()
    plt.plot(val_plot, color='green', marker='o', label='val_'+label_name)
    plt.title('model '+label_name)
    plt.xlabel('epoch')
    plt.ylabel(label_name)
    plt.legend()
    fig.savefig(save_path + file_name)


###########################
# 4.モデルによる予測
###########################

class ModelTest_CLOTH():
    def __init__(self, model, maxlen_words, word_to_id, vec_dict, ft_path, bin_path, id_to_word):
        self.model=model
        self.N=maxlen_words
        self.word_to_id=word_to_id
        self.vec_dict=vec_dict
        self.ft_path=ft_path
        self.bin_path=bin_path
        self.id_to_word=id_to_word


    def preprocess_for_test(self, s):
        sent_tokens=[]
        s = unicodeToAscii(s)
        s = re.sub(r'[ ]+', ' ', s)
        s = s.strip()
        tokens=nltk.word_tokenize(s)
        symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
        num_tag=("LS", "CD")
        tagged = nltk.pos_tag(tokens)
        for word, tag in tagged:
            if word==CLZ_word:
                sent_tokens.append(CLZ_word)
            if tag in symbol_tag:
                pass
                #記号は無視
            elif tag in num_tag:
                sent_tokens.append('NUM')
            else:
                sent_tokens.append(word.lower())

        return sent_tokens


    #nltkのtoken列をidsへ
    def token_to_ids_for_test(self, tokens):
        ids=[]

        for word in tokens:
            ids.append(search_word_indices(word, self.word_to_id))

        len_ids=len(ids)

        if len_ids<self.N:
            return [0] * (self.N - len_ids) +ids
        else:
            return ids[:self.N]


    #選択肢が全て1語かどうかのチェック
    def is_one_word(self, choices):
        for c in choices:
            if c.count(' ')>0:
                return False

        return True


    #2つのベクトルのコサイン類似度を返す
    def calc_similarity(self, pred_vec, ans_vec):
        len_p=np.linalg.norm(pred_vec)
        len_a=np.linalg.norm(ans_vec)
        if len_p==0 or len_a==0:
            return 0.0
        return np.dot(pred_vec/len_p, ans_vec/len_a)


    #直近予測スコアの算出
    #モデルの出力と各選択肢との類似度
    def calc_near_scores(self, cloze_sent, choices):
        scores=[]
        tokens=self.preprocess_for_test(cloze_sent)
        clz_index=tokens.index(CLZ_word)

        before=tokens[:clz_index]
        after=tokens[clz_index+1:]

        f_X=self.token_to_ids_for_test(before)
        r_X=self.token_to_ids_for_test(after[::-1])

        preds_vec = self.model.predict([[f_X], [r_X]], verbose=0)

        #choices は必ず1語
        for word in choices:
            word_vec=get_ft_vec(word, self.vec_dict, self.ft_path, self.bin_path)
            score=self.calc_similarity(preds_vec, word_vec)
            scores.append(score)

        return scores


    def make_inputs_for_sent_score(self, cloze_sent, choice_words):
        sent=cloze_sent.replace(CLZ_word, choice_words)
        tokens=self.preprocess_for_test(sent)

        ids=[]
        vecs=[]

        len_text=len(tokens)

        for word in tokens:
            ids.append(search_word_indices(word, self.word_to_id))
            vecs.append(get_ft_vec(word, self.vec_dict, self.ft_path, self.bin_path))

        while(len_text < self.N*2+1):
            ids=[0]+ids+[0]
            vecs.insert(0, np.zeros(vec_size))
            vecs.append(np.zeros(vec_size))
            len_text+=2

        return ids, vecs


    #補充文スコアの算出
    def calc_sent_scores(self, cloze_sent, choices):
        scores=[]

        for words in choices:
            score=0
            ids, vecs=self.make_inputs_for_sent_score(cloze_sent, words)

            sent_len=len(ids)

            #Nとかの数rangeのとこも要確認
            for i in range(sent_len-2*self.N-1):
                #長さ計って、iとかでforループ？
                f_X=ids[i : i+self.N]
                r_X=ids[i+self.N+1 : i+2*self.N+1]

                word_vec=vecs[i+self.N]
                preds_vec = self.model.predict([[f_X], [r_X]], verbose=0)
                tmp_score=self.calc_similarity(preds_vec, word_vec)

                if not tmp_score>0:
                    tmp_score=0.00000001  #仮
                score+=math.log(tmp_score)

            scores.append(score/(sent_len-2*self.N))

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

        for cloze_sent, choices, ans_word in zip(cloze_list, choices_list, ans_list):
            #直近予測スコア(1語のみ)
            line, OK=self.check_one_sent_by_near_score(cloze_sent, choices, ans_word)
            near_line+=line
            near_OK+=OK

            #補充文スコア(1語のみ)
            line, OK=self.check_one_sent_by_sent_score(cloze_sent, choices, ans_word, one_word=True)
            sent_line_one_word+=line
            sent_OK_one_word+=OK

            #補充文スコア(1語以上)
            line, OK=self.check_one_sent_by_sent_score(cloze_sent, choices, ans_word, one_word=False)
            sent_line+=line
            sent_OK+=OK

        print('near score')
        print('line:%d, acc:%.4f'% (near_line, 1.0*near_OK/near_line))

        print('sent score (one word)')
        print('line:%d, acc:%.4f'% (sent_line_one_word, 1.0*sent_OK_one_word/sent_line_one_word))

        print('sent score')
        print('line:%d, acc:%.4f'% (sent_line, 1.0*sent_OK/sent_line))



#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test', 'train_loop'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--model', help='model file name (when load model, mode=test)')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--model_kind', choices=['origin', 'plus_CAR', 'plus_KenLM', 'plus_both', 'replace_CAR', 'replace_KenLM', 'replace_both'], default='origin', help='model file kind')

    # ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()
    print(args.mode)
    epoch=args.epoch


    # 2.fasttextのロードと辞書の作成
    '''
    https://github.com/facebookresearch/fastText
    このfastextを事前に実行しておき，その結果を利用
    '''

    ft_path='../../../../M1/FastText/fastText-0.1.0/fasttext'

    #ベクトルファイル
    vec_path='../../../../M1/FastText/Model/text8_dim'+str(vec_size)+'_minC0.vec'
    bin_path='../../../../M1/FastText/Model/text8_dim'+str(vec_size)+'_minC0.bin'

    len_words, word_to_id, id_to_word, vec_dict=vec_to_dict(vec_path)

    weights_matrix = np.zeros((len_words+1, vec_size))
    if args.mode == 'all':
        for i in range(len_words):
            if i!=0:
                #IDが0の単語が存在しないので0は飛ばす
                weights_matrix[i] = get_ft_vec(id_to_word[i], vec_dict, ft_path, bin_path)

    #通常時
    # 2.モデル定義
    model = build_model(len_words, weights_matrix)

    #学習時
    if args.mode == 'all' or args.mode == 'mini':
        train_path=CLOTH_path+'for_KenLM_CLOTH.txt'
        val_path=CLOTH_path+'for_KenLM_CLOTH_val.txt'

        #モデルとか結果とかを格納するディレクトリの作成
        save_path=save_path+'_BiVecPresModel'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'
        #plot_model(model, to_file=save_path+'BiVecPresModel.png', show_shapes=True)
        #model.summary()

        # 3.学習
        model = trainIters(model, train_path, val_path, len_words, word_to_id, id_to_word, vec_dict, ft_path, bin_path, n_iters=epoch, saveModel=True)
        print('Train end')

    #すでにあるモデルでテスト時
    else:
        save_path=args.model_dir+'/'
        model.load_weights(save_path+args.model+'.hdf5')
        #model.summary()

        save_path=save_path+today_str

    # テストの実行
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

    test=ModelTest_CLOTH(model, maxlen_words, word_to_id, vec_dict, ft_path, bin_path, id_to_word)

    for data in datas:
        data_name=data[0]
        cloze_path=data[1]
        choices_path=data[2]
        ans_path=data[3]

        test.model_test_both_score(data_name, cloze_path, choices_path, ans_path)

    end_test=print_time('test end')
