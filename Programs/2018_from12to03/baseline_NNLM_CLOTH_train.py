# -*- coding: utf-8 -*-

'''
ベースライン用のNNLM
seq2seqの時はpytorchで書いてたけど、kerasで書きなおしたやつ
構造は同じはず

CLOTHで学習、CLOTHと同じ前処理

#TODO まだ未作成、まだ何も手つけてない

動かしていたバージョン
python  : 3.5.2

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


maxlen_words = 5
BATCH_SIZE=256

PAD_token = 0
UNK_token = 1
NUM_token = 2

file_path='../../../pytorch_data/'
git_data_path='../../Data/'
CLOTH_path = file_path+'CLOTH_for_model/'

today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str


#事前処理いろいろ
print('Start: '+today_str)

#----- 関数群 -----

###########################
# 1.データの準備，データ変換
###########################

#seq2seqモデルで用いる語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": UNK_token}
        self.word2count = {"<UNK>": 0}
        self.index2word = {PAD_token: "PAD", UNK_token: "<UNK>", NUM_token: "NUM"}
        self.n_words = 3  # PAD と SOS と EOS と UNK　とNUM

    #文から単語を語彙へ
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    #語彙のカウント
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def check_word2index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["<UNK>"]


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


#与えた語彙読み込み
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(line.strip())
    #print("Vocab: %s" % lang.n_words)

    return lang

#nltkのtoken列をidsへ
def token_to_ids(lang, tokens, maxlen):
    ids=[]
    #NLTKの記号を表すタグ
    symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
    #NLTKの数詞を表すタグ
    num_tag=("LS", "CD")
    #他のNLTKタグについては nltk.help.upenn_tagset()
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if word==CLZ_word:
            ids.append(CLZ_token)
        elif tag in symbol_tag:
            pass
            #記号は無視
        elif tag in num_tag:
            ids.append(NUM_token)
        else:
            ids.append(lang.check_word2index(word.lower()))

    return ids + [PAD_token] * (maxlen - len(ids))


#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


#空所つき1文をidsへ
def sent_to_ids_cloze(lang, s):
    # s は文字列
    s = unicodeToAscii(s)
    s = re.sub(r'[ ]+', ' ', s)
    s = s.strip()
    s = re.sub(r'{.+}', CLZ_word, s)
    token=nltk.word_tokenize(s)
    ids=token_to_ids(lang, token, MAX_LENGTH)

    return ids


#選択肢4つをidsへ
def choices_to_ids(lang, choices):
    # choices は文字列のリスト
    ids=[]
    for choi in choices:
        choi = unicodeToAscii(choi)
        choi = re.sub(r'[ ]+', ' ', choi)
        choi = choi.strip()
        token=nltk.word_tokenize(choi)
        id=token_to_ids(lang, token, C_MAXLEN)
        ids.append(id)

    #デバッグ時確認用
    if len(ids)!=4:
        print('### choices_to_ids ERROR')
        print(choices)
        exit()

    return ids


#正答一つをidsへ
def ans_to_ids(lang, ans, choices):
    # ans は文字列
    # choices は文字列のリスト
    ids = [1 if choi==ans else 0 for choi in choices]

    #デバッグ時確認用
    if sum(ids)!=1:
        print('### ans_to_ids ERROR')
        print(ids)
        print(ans)
        print(choices)
        exit()

    return ids


#Googleのword2vec読み取り
def get_weight_matrix(lang):
    print('Loading word vector ...')
    #ここのgensimの書き方がバージョンによって異なる
    vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_path+'GoogleNews-vectors-negative300.bin', binary=True)
    # https://code.google.com/archive/p/word2vec/ ここからダウンロード&解凍

    weights_matrix = np.zeros((lang.n_words, EMB_DIM))

    for i, word in lang.index2word.items():
        try:
            weights_matrix[i] = vec_model.wv[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(EMB_DIM, ))

    del vec_model
    #これメモリ解放的なことらしい、なくてもいいかも

    #パディングのところを初期化
    #Emneddingで引数のpad_index指定は、そこだけ更新(微分)しないらしい？
    weights_matrix[PAD_token]=np.zeros(EMB_DIM)

    return weights_matrix



###########################
# 2.モデル定義
###########################

# モデルの構築
def build_model(lang, embedding_matrix):
    input=Input(shape=(maxlen_words,))
    emb=Embedding(output_dim=300, input_dim=lang.n_words, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=Ture)(input)

    emb=Dropout(0.5)(emb)

    lstm1=Bidirectional(LSTM(128, dropout=0.5, return_sequences=True))(emb)
    lstm_out=Bidirectional(LSTM(128, dropout=0.5, return_sequences=False))(lstm1)

    output=Dense(lang.n_words)(lstm_out)
    output=Activation('softmax')(output)

    my_model=Model(input, output)

    optimizer = RMSprop()
    my_model.compile(loss='categorical_crossentropy', optimizer=optimizer)

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
def tokens_to_data(tokens, lang):
    X = []
    Y = []
    len_text=len(tokens)
    leng=0

    ids=[]
    for word in tokens:
        ids.append(lang.check_word2index(word))


    len_text=len(ids)

    while(len_text < maxlen_words+1):
        ids=[0]+ids+[0]
        len_text+=2

    for i in range(len_text - maxlen_words -1):
        x=ids[i: i + maxlen_words]
        n=ids[i + maxlen_words]
        y=np.zeros(lang.n_words)
        y[n]=1
        X.append(x)
        Y.append(y)

    return X, Y


#空所等を含まない英文のデータから，モデルの入出力を作成
def make_data(file_path, lang):
    X_list=[]
    Y_list=[]

    with open(file_path, encoding='utf-8') as f:
        for line in f:
            tokens=preprocess(line)
            x, y=tokens_to_data(tokens, lang)
            X_list.extend(x)
            Y_list.extend(y)
    X=np.array(X_list, dtype=np.int)
    Y=np.array(Y_list, dtype=np.int)

    return X, Y


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
def trainIters(model, train_path, val_path, lang, n_iters=5, print_every=10, saveModel=False):

    print('Make data for model...')
    X_train, Y_train=make_data(train_path, lang)
    X_val, Y_val=make_data(val_path, lang)


    X_train=X_train
    X_val=X_val

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

        for cloze_sent, choices, ans_word in zip(cloze_list, choices_list, ans_list):
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

    # 1.語彙データ読み込み
    vocab_path=file_path+'enwiki_vocab30000_wordonly.txt'
    vocab = readVocab(vocab_path)

    if args.mode == 'all' or args.mode == 'train_loop':
        weights_matrix = get_weight_matrix(vocab)
    else:
        weights_matrix = np.zeros((vocab.n_words, EMB_DIM))


    #通常時
    # 2.モデル定義
    model = build_model(vocab, weights_matrix)

    #学習時
    if args.mode == 'all' or args.mode == 'mini':
        train_path=CLOTH_path+'for_KenLM_CLOTH.txt'
        val_path=CLOTH_path+'for_KenLM_CLOTH_val.txt'

        #モデルとか結果とかを格納するディレクトリの作成
        save_path=save_path+'_NNLM'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'
        #plot_model(model, to_file=save_path+'BiVecPresModel.png', show_shapes=True)
        #model.summary()

        # 3.学習
        model = trainIters(model, train_path, val_path, vocab, n_iters=epoch, saveModel=True)
        print('Train end')
        exit()
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

    test=ModelTest(model, maxlen_words, word_to_id, vec_dict, ft_path, bin_path, id_to_word)

    for data in datas:
        data_name=data[0]
        cloze_path=data[1]
        choices_path=data[2]
        ans_path=data[3]

        test.model_test_both_score(data_name, cloze_path, choices_path, ans_path)

    end_test=print_time('test end')
