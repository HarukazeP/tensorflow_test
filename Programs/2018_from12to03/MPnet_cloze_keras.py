# -*- coding: utf-8 -*-

'''
MPNet_cloze.py から変更
pytorchで実装してたけど解決できないエラーに直面したためkerasで書き直し



Liang Wangらnoの論文
「Multi-Perspective Context Aggregation for Semi-supervised Cloze-style Reading Comprehension」
ttps://aclanthology.coli.uni-saarland.de/papers/C18-1073/c18-1073
を読んでモデル自分なりにできる範囲で真似したやつ

論文のモデル厳密には再現できていない
半教師あり学習は未実装

----- 論文に書いてたこと -----
!! は再現できていないところ

!! ハイパーパラメータは検証データでのランダムサーチにより決定
!! embeddingの初期値はGloveの300次元，上位1000単語のみ学習で更新   # memo:これってもしかして1000語のEmbeddingしてる？
最適化関数はadam
!! 学習率は最初10^(-3)，学習を重ねるごとに10^(-4)や10^(-5)に小さくしていく
gradient clipingの設定は最大値√5
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，フィルター数128，width3
GRUのドロップアウト率は50%


----- 自分の実装 -----

!! はもとの論文と異なる点
それ以外にも，次元数というか階数が元の論文あまり書いてなかったから割と違うかも

!! 語彙数は上位3万単語
!! embeddingの初期値はGloveの300次元(glove.6B.300d.txt)，全て学習で更新
最適化関数はadam
!! 学習率は10^(-3)
gradient clipingの設定は最大値√5
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，出力次元数128，kernel_size 3
!! CNNのバッチ正規化とかReLUの利用場所も曖昧
GRUのドロップアウト率は50%


動かしていたバージョン
python  : 3.5.2 / 3.6.5
keras   : 0.2.0
gensim  : 3.1.0 / 3.5.0


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

import gensim
import nltk

from keras import regularizers
from keras import backend as K
from keras.models import Model, model_from_json, load_model
from keras.layers import Dense, Embedding, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D
from keras.layers import Add, Multiply, Input, Activation, Reshape, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from keras.utils.vis_utils import plot_model
from keras.activations import softmax, sigmoid
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from glob import glob
import kenlm

#----- グローバル変数一覧 -----
MAX_LENGTH = 80
C_MAXLEN = 6
HIDDEN_DIM = 128
EMB_DIM = 300
BATCH_SIZE = 128

file_path='../../../pytorch_data/'
git_data_path='../../Data/'
CLOTH_path = file_path+'CLOTH_for_model/'
KenLM_path='/media/tamaki/HDCL-UT/tamaki/M2/kenlm_models/all_words/'

today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str
PAD_token = 0
UNK_token = 1
CLZ_token = 2
NUM_token = 3

CLZ_word = 'XXXX'

#事前処理いろいろ
print('Start: '+today_str)


#----- 関数群 -----

###########################
# 1.データの準備，データ変換
###########################

#語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": UNK_token}
        self.index2word = {PAD_token: "PAD", UNK_token: "<UNK>", CLZ_token: "CLZ", NUM_token: "NUM"}
        self.n_words = 4  # PAD, UNK, CLZ, NUM

    #文から単語を語彙へ
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    #語彙のカウント
    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.index2word[self.n_words] = word
            self.n_words += 1

    def check_word2index(self, word):
        if word in self.word2index:
            return self.word2index[word]
        else:
            return self.word2index["<UNK>"]


#与えた語彙読み込み
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(line.strip())
    #print("Vocab: %s" % lang.n_words)

    return lang


#空所つき英文読み取り
def readCloze(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
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
    '''
    #ここのgensimの書き方がバージョンによって異なる
    vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_path+'GoogleNews-vectors-negative300.bin', binary=True)
    # ttps://code.google.com/archive/p/word2vec/ ここからダウンロード&解凍
    '''


    '''
    Gloveのベクトル使用
    ttps://nlp.stanford.edu/projects/glove/

    事前に↓これしておく
    from gensim.scripts.glove2word2vec import glove2word2vec
    glove2word2vec(glove_input_file="glove.6B.300d.txt", word2vec_output_file="gensim_glove_vectors.txt")
    '''
    vec_model =gensim.models.KeyedVectors.load_word2vec_format(file_path+'gensim_glove_vectors.txt', binary=False)

    weights_matrix = np.zeros((lang.n_words, EMB_DIM))

    for i, word in lang.index2word.items():
        try:
            weights_matrix[i] = vec_model.wv[word] #gensimのword2vec

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

# Reshape処理とかの前にこれを挟まないとエラーでる
'''
Layer reshape_1 does not support masking, but was passed an input_mask
ttps://github.com/keras-team/keras/issues/4978
'''
class NonMasking(Layer):
    def __init__(self, **kwargs):
        self.supports_masking = True
        super(NonMasking, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        return x

    def get_output_shape_for(self, input_shape):
        return input_shape


#自作レイヤー Selective Copying 用
class SCLayer(Layer):
    def __init__(self, output_dim, bsize, **kwargs):
        self.output_dim = output_dim
        self.bs = bsize
        super(SCLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        input_shape = input_shape

    def call(self, inputs):
        '''
        inputs=[sent_vec, cloze_input]
        引数一つしか無理らしいのでリストにしてる
        '''
        sent_vec, cloze_input=inputs

        cloze_vec=K.batch_dot(sent_vec, cloze_input, axes=[1,1]) # (b, h)

        return cloze_vec

    def compute_output_shape(self, input_shape):
        bs=self.bs
        h=self.output_dim
        return (bs, h)


#自作レイヤー Attentive Reader用
class ARLayer(Layer):
    def __init__(self, output_dim, bsize, **kwargs):
        self.output_dim = output_dim
        self.bs = bsize
        super(ARLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,1),
                                    initializer='uniform',          trainable=True)
        super(ARLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):
        '''
        inputs=[sent_vec, c1_vec, c2_vec, c3_vec, c4_vec]
        引数一つしか無理らしいのでリストにしてる
        '''
        sent_vec, c1_vec, c2_vec, c3_vec, c4_vec=inputs

        Wh=K.dot(sent_vec, self.kernel)  # (b, s, 2h)

        bh=K.dot(sent_vec, self.bias) # (b, s, 1)

        u1=K.expand_dims(c1_vec, axis=2) # (b, 2h) -> (b, 2h, 1)
        u2=K.expand_dims(c2_vec, axis=2)
        u3=K.expand_dims(c3_vec, axis=2)
        u4=K.expand_dims(c4_vec, axis=2)

        u1_Wh=K.batch_dot(Wh, u1, axes=[2,1]) # (b, s, 1)
        u2_Wh=K.batch_dot(Wh, u2, axes=[2,1])
        u3_Wh=K.batch_dot(Wh, u3, axes=[2,1])
        u4_Wh=K.batch_dot(Wh, u4, axes=[2,1])

        attn_1=softmax(u1_Wh+bh, axis=1) # (b, s, 1)
        attn_2=softmax(u2_Wh+bh, axis=1)
        attn_3=softmax(u3_Wh+bh, axis=1)
        attn_4=softmax(u4_Wh+bh, axis=1)

        attn1_h=sent_vec*attn_1  # (b, s, 2h)
        attn2_h=sent_vec*attn_2
        attn3_h=sent_vec*attn_3
        attn4_h=sent_vec*attn_4

        P1=K.sum(attn1_h, axis=1)    # (b, s, 2h) -> (b, 2h)
        P2=K.sum(attn2_h, axis=1)
        P3=K.sum(attn3_h, axis=1)
        P4=K.sum(attn4_h, axis=1)


        return [P1, P2, P3, P4]

    def compute_output_shape(self, input_shape):
        bs=self.bs
        h=self.output_dim
        return [(bs, h), (bs, h), (bs, h), (bs, h)]


#自作レイヤー Attentive Reader用
class CARLayer(Layer):
    def __init__(self, output_dim, bsize, **kwargs):
        self.output_dim = output_dim
        self.bs = bsize
        super(CARLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel',
                                      shape=(self.output_dim, self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        self.bias = self.add_weight(name='bias',
                                    shape=(self.output_dim,1),
                                    initializer='uniform',          trainable=True)
        super(CARLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        CAR_sent1_vec, CAR_sent2_vec, CAR_sent3_vec, CAR_sent4_vec, c1_vec, c2_vec, c3_vec, c4_vec=inputs

        Wh1=K.dot(CAR_sent1_vec, self.kernel)  # (b, s, 2h)
        Wh2=K.dot(CAR_sent2_vec, self.kernel)
        Wh3=K.dot(CAR_sent3_vec, self.kernel)
        Wh4=K.dot(CAR_sent4_vec, self.kernel)

        bh1=K.dot(CAR_sent1_vec, self.bias) # (b, s, 1)
        bh2=K.dot(CAR_sent2_vec, self.bias)
        bh3=K.dot(CAR_sent3_vec, self.bias)
        bh4=K.dot(CAR_sent4_vec, self.bias)

        u1=K.expand_dims(c1_vec, axis=2) # (b, 2h) -> (b, 2h, 1)
        u2=K.expand_dims(c2_vec, axis=2)
        u3=K.expand_dims(c3_vec, axis=2)
        u4=K.expand_dims(c4_vec, axis=2)

        u1_Wh1=K.batch_dot(Wh1, u1, axes=[2,1]) # (b, s, 1)
        u2_Wh2=K.batch_dot(Wh2, u2, axes=[2,1])
        u3_Wh3=K.batch_dot(Wh3, u3, axes=[2,1])
        u4_Wh4=K.batch_dot(Wh4, u4, axes=[2,1])

        attn_1=softmax(u1_Wh1+bh1, axis=1) # (b, s, 1)
        attn_2=softmax(u2_Wh2+bh2, axis=1)
        attn_3=softmax(u3_Wh3+bh3, axis=1)
        attn_4=softmax(u4_Wh4+bh4, axis=1)

        attn1_h=CAR_sent1_vec*attn_1  # (b, s, 2h)
        attn2_h=CAR_sent2_vec*attn_2
        attn3_h=CAR_sent3_vec*attn_3
        attn4_h=CAR_sent4_vec*attn_4

        P1=K.sum(attn1_h, axis=1)    # (b, s, 2h) -> (b, 2h)
        P2=K.sum(attn2_h, axis=1)
        P3=K.sum(attn3_h, axis=1)
        P4=K.sum(attn4_h, axis=1)


        return [P1, P2, P3, P4]

    def compute_output_shape(self, input_shape):
        bs=self.bs
        h=self.output_dim
        return [(bs, h), (bs, h), (bs, h), (bs, h)]


#自作レイヤー 出力層用
class PointerNet(Layer):
    def __init__(self, output_dim, Pdim, Cdim, bsize, **kwargs):
        self.output_dim = output_dim
        self.choices_num = 4 #選択肢の数
        self.P_hidden=Pdim
        self.C_hidden=Cdim
        self.bs=bsize

        super(PointerNet, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.GateWeight_P = self.add_weight(name='gateWP',
                                      shape=(self.P_hidden, self.C_hidden),
                                      initializer='uniform',
                                      trainable=True)
        self.GateWeight_C = self.add_weight(name='gateWC',
                                      shape=(self.C_hidden, self.C_hidden),
                                      initializer='uniform',
                                      trainable=True)
        self.GateBias = self.add_weight(shape=(self.C_hidden,),
                                    name='gatebias',
                                    initializer='uniform',
                                    trainable=True)
        self.OutWeight = self.add_weight(name='outW',
                                      shape=(self.P_hidden, self.C_hidden),
                                      initializer='uniform',
                                      trainable=True)
        self.OutBias = self.add_weight(name='outbias',
                                    shape=(self.C_hidden,1),
                                    initializer='uniform',
                                    trainable=True)
        super(PointerNet, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, inputs):

        P, C1, C2, C3, C4 = inputs

        WP=K.dot(P, self.GateWeight_P)  # (b, 2h) -> (b, 4h)

        WC1=K.dot(C1, self.GateWeight_C)    # (b, 4h) -> (b, 4h)
        WC2=K.dot(C2, self.GateWeight_C)
        WC3=K.dot(C3, self.GateWeight_C)
        WC4=K.dot(C4, self.GateWeight_C)

        g1=sigmoid(WP+WC1+self.GateBias) #(b, 4h)　#これでちゃんとバッチ数分バイス足せてる
        g2=sigmoid(WP+WC2+self.GateBias)
        g3=sigmoid(WP+WC3+self.GateBias)
        g4=sigmoid(WP+WC4+self.GateBias)

        C_dash1=C1*g1   #(b, 4h)
        C_dash2=C2*g2
        C_dash3=C3*g3
        C_dash4=C4*g4

        WP_out=K.dot(P, self.OutWeight)   # (b, 2h) -> (b, 4h)

        C1WP=K.batch_dot(C_dash1, WP_out, axes=[1,1]) # (b, 1)
        C2WP=K.batch_dot(C_dash2, WP_out, axes=[1,1])
        C3WP=K.batch_dot(C_dash3, WP_out, axes=[1,1])
        C4WP=K.batch_dot(C_dash4, WP_out, axes=[1,1])

        bC1=K.dot(C_dash1, self.OutBias)    # (b, 1)
        bC2=K.dot(C_dash2, self.OutBias)
        bC3=K.dot(C_dash3, self.OutBias)
        bC4=K.dot(C_dash4, self.OutBias)

        out1=C1WP+bC1   #(b,1)
        out2=C2WP+bC2
        out3=C3WP+bC3
        out4=C4WP+bC4

        output=K.concatenate([out1, out2, out3, out4], axis=1)   #(b,4)

        return output

    def compute_output_shape(self, input_shape):
        return (self.bs, self.choices_num)


def build_model(vocab_size, emb_size, hidden_size, emb_matrix, my_model_kind):
    use_Ng, use_AR, use_KenLM, use_CAR=use_config(my_model_kind)
    # --- 論文中のInput Layer ---
    sent_input=Input(shape=(MAX_LENGTH,))   #(b, s)
    c1=Input(shape=(C_MAXLEN,)) #(b, c)
    c2=Input(shape=(C_MAXLEN,))
    c3=Input(shape=(C_MAXLEN,))
    c4=Input(shape=(C_MAXLEN,))

    sent_E=Embedding(output_dim=emb_size, input_dim=vocab_size, input_length=MAX_LENGTH, mask_zero=True, weights=[emb_matrix], trainable=True)
    sent_emb=sent_E(sent_input)

    choices_E=Embedding(output_dim=emb_size, input_dim=vocab_size, input_length=C_MAXLEN, mask_zero=True, weights=[emb_matrix], trainable=True)
    c1_emb=choices_E(c1)    #(b, c, h)
    c2_emb=choices_E(c2)
    c3_emb=choices_E(c3)
    c4_emb=choices_E(c4)

    sent_vec=Bidirectional(GRU(hidden_size, dropout=0.5, return_sequences=True))(sent_emb) #(b, s, 2h)

    choices_BiGRU=Bidirectional(GRU(hidden_size, dropout=0.5, return_sequences=True))
    c1_gru=NonMasking()(choices_BiGRU(c1_emb))    #(b, c, 2h)
    c2_gru=NonMasking()(choices_BiGRU(c2_emb))
    c3_gru=NonMasking()(choices_BiGRU(c3_emb))
    c4_gru=NonMasking()(choices_BiGRU(c4_emb))

    c1_vec=Reshape((hidden_size*2*C_MAXLEN,))(c1_gru)    #(b, c*2h)
    c2_vec=Reshape((hidden_size*2*C_MAXLEN,))(c2_gru)
    c3_vec=Reshape((hidden_size*2*C_MAXLEN,))(c3_gru)
    c4_vec=Reshape((hidden_size*2*C_MAXLEN,))(c4_gru)

    choices_Dense=Dense(hidden_size*2)
    c1_vec=choices_Dense(c1_vec)    #(b, 2h)
    c2_vec=choices_Dense(c2_vec)
    c3_vec=choices_Dense(c3_vec)
    c4_vec=choices_Dense(c4_vec)

    # --- 論文中のMulti-Perspective Aggregation Layer ---
    bsize=K.int_shape(sent_vec)[0]

    # --- MPALayerの一部: Selective Copying ---
    cloze_input=Input(shape=(MAX_LENGTH,))   #(b, s)
    P_sc = SCLayer(hidden_size*2, bsize)([NonMasking()(sent_vec), NonMasking()(cloze_input)])

    # --- MPALayerの一部: Iterative Dilated Convolution ---
    sent_cnn = BatchNormalization(axis=2)(sent_vec)
    sent_cnn = Activation("relu")(sent_cnn)
    sent_cnn = NonMasking()(sent_cnn)
    sent_cnn = Conv1D(hidden_size*2, kernel_size=3, dilation_rate=1)(sent_cnn)
    sent_cnn = Conv1D(hidden_size*2, kernel_size=3, dilation_rate=3)(sent_cnn)
    #sent_cnn = BatchNormalization(axis=2)(sent_cnn)
    #sent_cnn = Activation("relu")(sent_cnn)
    sent_cnn = Conv1D(hidden_size*2, kernel_size=3, dilation_rate=1)(sent_cnn)

    sent_cnn = Conv1D(hidden_size*2, kernel_size=3, dilation_rate=3)(sent_cnn)
    P_idc = GlobalMaxPooling1D()(sent_cnn)

    # --- MPALayerの一部: Attentive Reader ---
    if use_AR==1:
        P1_ar, P2_ar, P3_ar, P4_ar=ARLayer(hidden_size*2, bsize)([NonMasking()(sent_vec), c1_vec, c2_vec, c3_vec, c4_vec])

    # --- MPALayerの一部: N-gram Statistics ---
    if use_Ng==1:
        Ngram_1=Input(shape=(5,))   #(b, 5)
        Ngram_2=Input(shape=(5,))
        Ngram_3=Input(shape=(5,))
        Ngram_4=Input(shape=(5,))

        P1_ng = NonMasking()(Ngram_1)
        P2_ng = NonMasking()(Ngram_2)
        P3_ng = NonMasking()(Ngram_3)
        P4_ng = NonMasking()(Ngram_4)

    # 自作拡張: 空所補充文Attentive Reader
    if use_CAR==1:
        CAR_sent1=Input(shape=(MAX_LENGTH,))
        CAR_sent2=Input(shape=(MAX_LENGTH,))
        CAR_sent3=Input(shape=(MAX_LENGTH,))
        CAR_sent4=Input(shape=(MAX_LENGTH,))

        CAR_sent1_emb=sent_E(CAR_sent1)
        CAR_sent2_emb=sent_E(CAR_sent2)
        CAR_sent3_emb=sent_E(CAR_sent3)
        CAR_sent4_emb=sent_E(CAR_sent4)

        CAR_sent_GRU=Bidirectional(GRU(hidden_size, dropout=0.5, return_sequences=True))

        CAR_sent1_vec=NonMasking()(CAR_sent_GRU(CAR_sent1_emb)) #(b, s, 2h)
        CAR_sent2_vec=NonMasking()(CAR_sent_GRU(CAR_sent2_emb))
        CAR_sent3_vec=NonMasking()(CAR_sent_GRU(CAR_sent3_emb))
        CAR_sent4_vec=NonMasking()(CAR_sent_GRU(CAR_sent4_emb))

        P1_car, P2_car, P3_car, P4_car=CARLayer(hidden_size*2, bsize)([CAR_sent1_vec, CAR_sent2_vec, CAR_sent3_vec, CAR_sent4_vec, c1_vec, c2_vec, c3_vec, c4_vec])

    # 自作拡張: KenLM Score
    if use_KenLM==1:
        KenLM_1=Input(shape=(5,))   #(b, 5)
        KenLM_2=Input(shape=(5,))
        KenLM_3=Input(shape=(5,))
        KenLM_4=Input(shape=(5,))

        P1_ks = NonMasking()(KenLM_1)
        P2_ks = NonMasking()(KenLM_2)
        P3_ks = NonMasking()(KenLM_3)
        P4_ks = NonMasking()(KenLM_4)

    # --- MPALayerの一部: 最後にマージ ---
    P =  Concatenate(axis=1)([P_sc, P_idc])     #(b, 2h+2h)

    C1_tmp=[c1_vec]
    C2_tmp=[c2_vec]
    C3_tmp=[c3_vec]
    C4_tmp=[c4_vec]

    if use_AR==1:
        C1_tmp.append(P1_ar)
        C2_tmp.append(P2_ar)
        C3_tmp.append(P3_ar)
        C4_tmp.append(P4_ar)

    if use_Ng==1:
        C1_tmp.append(P1_ng)
        C2_tmp.append(P2_ng)
        C3_tmp.append(P3_ng)
        C4_tmp.append(P4_ng)

    if use_CAR==1:
        C1_tmp.append(P1_car)
        C2_tmp.append(P2_car)
        C3_tmp.append(P3_car)
        C4_tmp.append(P4_car)

    if use_KenLM==1:
        C1_tmp.append(P1_ks)
        C2_tmp.append(P2_ks)
        C3_tmp.append(P3_ks)
        C4_tmp.append(P4_ks)

    C1 = Concatenate(axis=1)(C1_tmp)
    C2 = Concatenate(axis=1)(C2_tmp)
    C3 = Concatenate(axis=1)(C3_tmp)
    C4 = Concatenate(axis=1)(C4_tmp)

    # --- 論文中のOutput Layer (PointerNet) ---
    # 出力層一応完了
    Pdim=K.int_shape(P)[-1]
    Cdim=K.int_shape(C1)[-1]

    output=PointerNet(hidden_size*2, Pdim, Cdim, bsize)([P, C1, C2, C3, C4]) #(b, 4)
    #preds = softmax(output, axis=1)   #(b, 4)
    preds=Activation('softmax')(output)

    #--------------------------
    X=[sent_input, c1, c2, c3, c4, cloze_input]
    if use_Ng==1:
        X.extend([Ngram_1, Ngram_2, Ngram_3, Ngram_4])

    if use_CAR==1:
        X.extend([CAR_sent1, CAR_sent2, CAR_sent3, CAR_sent4])

    if use_KenLM==1:
        X.extend([KenLM_1, KenLM_2, KenLM_3, KenLM_4])

    my_model=Model(X, preds)
    opt=optimizers.Adam(lr=0.001, clipnorm=math.sqrt(5))    #デフォルト：lr=0.001
    my_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return my_model

###########################
# 3.モデルの学習
###########################

#checkpoint で保存された最新のモデル(ベストモデルをロード)
def getNewestModel(model):
    files = [(f, os.path.getmtime(f)) for f in glob(save_path+'*hdf5')]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


# 選択肢リストをモデルの入力の形へ分割
def split_choices(choices_array):
    c1, c2, c3, c4=np.split(choices_array, 4, axis=1)
    c1=np.squeeze(c1)
    c2=np.squeeze(c2)
    c3=np.squeeze(c3)
    c4=np.squeeze(c4)

    return c1, c2, c3, c4


# 空所の位置をone_hotで表すnumpy配列を返す
def make_cloze_onehot(X_tmp):
    cloze_onehot=np.zeros_like(X_tmp)
    cloze_onehot[X_tmp==CLZ_token]=1

    return cloze_onehot


class Ngram():
    def __init__(self):
        self.count_1gram = {}
        self.count_2gram = {}
        self.count_3gram = {}
        self.count_4gram = {}
        self.count_5gram = {}

        self.KenLM_1gram =''
        #kenlm.LanguageModel(KenLM_path+'CLOTH_NUM_N1.arpa')
        self.KenLM_2gram = kenlm.LanguageModel(KenLM_path+'CLOTH_NUM_N2.arpa')
        self.KenLM_3gram = kenlm.LanguageModel(KenLM_path+'CLOTH_NUM_N3.arpa')
        self.KenLM_4gram = kenlm.LanguageModel(KenLM_path+'CLOTH_NUM_N4.arpa')
        self.KenLM_5gram = kenlm.LanguageModel(KenLM_path+'CLOTH_NUM_N5.arpa')
    #前処理
    def preprocess(self, s):
        sent_tokens=[]
        s = unicodeToAscii(s)
        s = re.sub(r'[ ]+', ' ', s)
        s = s.strip()
        s = re.sub(r'{.+}', CLZ_word, s)
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


    #最初にngramのカウント用
    def count_ngram_first(self, tokens):
        dic=[self.count_1gram, self.count_2gram, self.count_3gram, self.count_4gram, self.count_5gram]
        for n in range(1,6):
            count_dic=dic[n-1]
            Ngr = nltk.ngrams(tokens, n)
            for gram in Ngr:
                if gram not in count_dic:
                    count_dic[gram]=1
                else:
                    count_dic[gram]+=1


    #最初にngramのカウント用
    def read_file_first(self, ans_file):
        sent=[]
        with open(ans_file, encoding='utf-8') as f:
            for s in f:
                s=re.sub(r'{ ', '', s)
                s=re.sub(r' }', '', s)
                tokens=self.preprocess(s)
                self.count_ngram_first(tokens)


    #空所に選択肢を補充した4文を生成
    def make_sents(self, cloze_sent, choices):
        sents=[]
        before=re.sub(r'{.*', '', cloze_sent)
        after=re.sub(r'.*}', '', cloze_sent)
        for choice in choices:
            tmp=before + choice + after
            tmp=tmp.strip()
            sents.append(tmp)
        return sents


    def sent_to_ngram_count(self, sent):
        ngram_count_sum_in_sent=[0, 0, 0, 0, 0]
        tokens=self.preprocess(sent)
        dic=[self.count_1gram, self.count_2gram, self.count_3gram, self.count_4gram, self.count_5gram]
        for n in range(1,6):
            count_dic=dic[n-1]
            Ngr = nltk.ngrams(tokens, n)
            for gram in Ngr:
                if gram  in count_dic:
                    ngram_count_sum_in_sent[n-1]+=count_dic[gram]

        return ngram_count_sum_in_sent


    def sent_to_KenLM_score(self, sent):
        KenLM_score=[0.0, 0.0, 0.0, 0.0, 0.0]
        tokens=self.preprocess(sent)
        kenlm_sent=' '.join(tokens)
        sent_len=len(tokens)

        #KenLM 1-gramだと使えないっぽいので飛ばしてる
        KenLM_models=[self.KenLM_1gram, self.KenLM_2gram, self.KenLM_3gram, self.KenLM_4gram, self.KenLM_5gram]
        for i in range(1,5):
            KenLM_score[i]=1.0*KenLM_models[i].score(kenlm_sent)/sent_len

        tmp_sum=0
        Ngr = nltk.ngrams(tokens, 1)
        for gram in Ngr:
            if gram  in self.count_1gram:
                tmp_sum+=self.count_1gram[gram]

        KenLM_score[0]=math.log(tmp_sum)

        return KenLM_score


    # ngram対数頻度のnumpy配列を返す
    def get_ngram_count(self, cloze_list, choices_list):
        '''
        引数：
            cloze_list    (問題数)
            choices_list  (問題数, 選択肢数4)
        返り値：
            ngram_count  （問題数，選択肢数4，ngram種類5）
        '''
        ngram_count=[]
        for cloze_sent, choices in zip(cloze_list, choices_list):
            s1, s2, s3, s4=self.make_sents(cloze_sent, choices)
            s1_count = self.sent_to_ngram_count(s1)
            s2_count = self.sent_to_ngram_count(s2)
            s3_count = self.sent_to_ngram_count(s3)
            s4_count = self.sent_to_ngram_count(s4)
            ngram_count.append([s1_count, s2_count, s3_count, s4_count])

        #対数頻度，log1p(x)はlog_{e}(x+1)を返す
        #頻度が0だとまずいので念のため+1してる
        log_count=np.log1p(np.array(ngram_count, dtype=np.float))

        n1, n2, n3, n4=np.split(log_count, 4, axis=1)
        n1=np.squeeze(n1)
        n2=np.squeeze(n2)
        n3=np.squeeze(n3)
        n4=np.squeeze(n4)
        return n1, n2, n3, n4


    def make_CAR_X(self, lang, cloze_list, choices_list):
        CAR_sent1=[]
        CAR_sent2=[]
        CAR_sent3=[]
        CAR_sent4=[]

        for cloze_sent, choices in zip(cloze_list, choices_list):
            s1, s2, s3, s4=self.make_sents(cloze_sent, choices)
            CAR_sent1.append(sent_to_ids_cloze(lang, s1))
            CAR_sent2.append(sent_to_ids_cloze(lang, s2))
            CAR_sent3.append(sent_to_ids_cloze(lang, s3))
            CAR_sent4.append(sent_to_ids_cloze(lang, s4))

        CAR1=np.array(CAR_sent1, dtype=np.int)
        CAR2=np.array(CAR_sent2, dtype=np.int)
        CAR3=np.array(CAR_sent3, dtype=np.int)
        CAR4=np.array(CAR_sent4, dtype=np.int)

        return CAR1, CAR2, CAR3, CAR4


    def get_KenLM_score(self, cloze_list, choices_list):
        KenLM_score=[]
        for cloze_sent, choices in zip(cloze_list, choices_list):
            s1, s2, s3, s4=self.make_sents(cloze_sent, choices)
            s1_score = self.sent_to_KenLM_score(s1)
            s2_score = self.sent_to_KenLM_score(s2)
            s3_score = self.sent_to_KenLM_score(s3)
            s4_score = self.sent_to_KenLM_score(s4)
            KenLM_score.append([s1_score, s2_score, s3_score, s4_score])

        #対数頻度，log1p(x)はlog_{e}(x+1)を返す
        #頻度が0だとまずいので念のため+1してる
        KenLM_array=np.array(KenLM_score, dtype=np.float)

        ks1, ks2, ks3, ks4=np.split(KenLM_array, 4, axis=1)
        ks1=np.squeeze(ks1)
        ks2=np.squeeze(ks2)
        ks3=np.squeeze(ks3)
        ks4=np.squeeze(ks4)
        return ks1, ks2, ks3, ks4


#学習をn_iters回，残り時間の算出をlossグラフの描画も
def trainIters(ngram, lang, model, train_pairs, val_pairs, my_model_kind, n_iters=5, print_every=10, learning_rate=0.001, saveModel=False):
    use_Ng, use_AR, use_KenLM, use_CAR=use_config(my_model_kind)

    X_train_tmp=np.array([sent_to_ids_cloze(lang, s) for s in train_pairs[0]], dtype=np.int)
    C_train=np.array([choices_to_ids(lang, s) for s in train_pairs[1]], dtype=np.int)
    Y_train=np.array([ans_to_ids(lang, s, c) for s,c in zip(train_pairs[2], train_pairs[1])], dtype=np.bool)

    X_val_tmp=np.array([sent_to_ids_cloze(lang, s) for s in val_pairs[0]], dtype=np.int)
    C_val=np.array([choices_to_ids(lang, s) for s in val_pairs[1]], dtype=np.int)
    Y_val=np.array([ans_to_ids(lang, s, c) for s,c in zip(val_pairs[2], val_pairs[1])], dtype=np.bool)

    print('train_ans_rate', np.sum(Y_train, axis=0))
    print('val_ans_rate', np.sum(Y_val, axis=0))

    c1_train, c2_train, c3_train, c4_train = split_choices(C_train)
    c1_val, c2_val, c3_val, c4_val = split_choices(C_val)

    # MPALayerの一部: Selective Copying 用の入力
    cloze_train=make_cloze_onehot(X_train_tmp)
    cloze_val=make_cloze_onehot(X_val_tmp)

    X_train=[X_train_tmp, c1_train, c2_train, c3_train, c4_train, cloze_train]
    X_val=[X_val_tmp, c1_val, c2_val, c3_val, c4_val, cloze_val]

    # MPALayerの一部: N-gram Statistics 用の入力
    if use_Ng==1:
        N1_train, N2_train, N3_train, N4_train=ngram.get_ngram_count(train_pairs[0], train_pairs[1])
        N1_val, N2_val, N3_val, N4_val=ngram.get_ngram_count(val_pairs[0], val_pairs[1])
        X_train.extend([N1_train, N2_train, N3_train, N4_train])
        X_val.extend([N1_val, N2_val, N3_val, N4_val])

    # 自作拡張: 空所補充文Attentive Reader 用の入力
    if use_CAR==1:
        CAR_sent1_train, CAR_sent2_train, CAR_sent3_train, CAR_sent4_train=ngram.make_CAR_X(lang, train_pairs[0], train_pairs[1])
        CAR_sent1_val, CAR_sent2_val, CAR_sent3_val, CAR_sent4_val=ngram.make_CAR_X(lang, val_pairs[0], val_pairs[1])

        X_train.extend([CAR_sent1_train, CAR_sent2_train, CAR_sent3_train, CAR_sent4_train])
        X_val.extend([CAR_sent1_val, CAR_sent2_val, CAR_sent3_val, CAR_sent4_val])


    # 自作拡張: KenLM Score 用の入力
    if use_KenLM==1:
        KenLM_1_train, KenLM_2_train, KenLM_3_train, KenLM_4_train=ngram.get_KenLM_score(train_pairs[0], train_pairs[1])
        KenLM_1_val, KenLM_2_val, KenLM_3_val, KenLM_4_val=ngram.get_KenLM_score(val_pairs[0], val_pairs[1])

        X_train.extend([KenLM_1_train, KenLM_2_train, KenLM_3_train, KenLM_4_train])
        X_val.extend([KenLM_1_val, KenLM_2_val, KenLM_3_val, KenLM_4_val])


    cp_cb = ModelCheckpoint(filepath = save_path+'model_ep{epoch:02d}.hdf5', monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    start = time.time()
    st_time=datetime.datetime.today().strftime('%H:%M')
    print("Training... ", st_time)


    # Ctrl+c で強制終了してもそこまでのモデルで残りの処理継続
    try:
        hist=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=n_iters, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb], shuffle=True)

        #全学習終わり
        #lossとaccのグラフ描画
        showPlot3(hist.history['loss'], hist.history['val_loss'], 'loss.png', 'loss')
        showPlot3(hist.history['acc'], hist.history['val_acc'], 'acc.png', 'acc')

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


def use_config(model_kind):
    use_Ng=0
    use_AR=0
    use_KenLM=0
    use_CAR=0
    if model_kind=='origin':
        use_Ng=1
        use_AR=1
    elif model_kind=='plus_CAR':
        use_Ng=1
        use_AR=1
        use_CAR=1
    elif model_kind=='plus_KenLM':
        use_Ng=1
        use_AR=1
        use_KenLM=1
    elif model_kind=='plus_both':
        use_Ng=1
        use_AR=1
        use_KenLM=1
        use_CAR=1
    elif model_kind=='replace_CAR':
        use_Ng=1
        use_CAR=1
    elif model_kind=='replace_KenLM':
        use_AR=1
        use_KenLM=1
    elif model_kind=='replace_both':
        use_KenLM=1
        use_CAR=1

    return use_Ng, use_AR, use_KenLM, use_CAR



###########################
# 4.モデルによる予測
###########################
def model_test(ngram, lang, model, cloze_path, choices_path, ans_path, my_model_kind, data_name='', file_output=True):
    print(data_name)
    use_Ng, use_AR, use_KenLM, use_CAR=use_config(my_model_kind)

    test_X=readCloze(cloze_path)
    test_C=readChoices(choices_path)
    test_Y=readAns(ans_path)

    if args.mode=='mini_test':
        test_X=test_X[:5]
        test_C=test_C[:5]
        test_Y=test_Y[:5]

    X_test_tmp=np.array([sent_to_ids_cloze(lang, s) for s in test_X], dtype=np.int)
    C_test=np.array([choices_to_ids(lang, s) for s in test_C], dtype=np.int)
    Y_test=np.array([ans_to_ids(lang, s, c) for s,c in zip(test_Y, test_C)], dtype=np.bool)

    print('test_ans_rate', np.sum(Y_test, axis=0))

    c1_test, c2_test, c3_test, c4_test = split_choices(C_test)

    # MPALayerの一部: Selective Copying 用の入力
    cloze_test=make_cloze_onehot(X_test_tmp)

    X_test=[X_test_tmp, c1_test, c2_test, c3_test, c4_test, cloze_test]

    # MPALayerの一部: N-gram Statistics 用の入力
    if use_Ng==1:
        N1_test, N2_test, N3_test, N4_test=ngram.get_ngram_count(test_X, test_C)
        X_test.extend([N1_test, N2_test, N3_test, N4_test])

    # 自作拡張: 空所補充文Attentive Reader 用の入力
    if use_CAR==1:
        CAR_sent1_test, CAR_sent2_test, CAR_sent3_test, CAR_sent4_test=ngram.make_CAR_X(lang, test_X, test_C)

        X_test.extend([CAR_sent1_test, CAR_sent2_test, CAR_sent3_test, CAR_sent4_test])


    # 自作拡張: KenLM Score 用の入力
    if use_KenLM==1:
        KenLM_1_test, KenLM_2_test, KenLM_3_test, KenLM_4_test=ngram.get_KenLM_score(test_X, test_C)

        X_test.extend([KenLM_1_test, KenLM_2_test, KenLM_3_test, KenLM_4_test])


    loss, acc=model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
    print('loss=%.4f, acc=%.4f' % (loss, acc))

    '''
    #モデルの中間層の出力確認用
    if args.mode=='mini_test':
        layers_names=['concatenate_1', 'concatenate_2', 'concatenate_3', 'concatenate_4', 'concatenate_5']
        for name in layers_names:
            out=Model(inputs=model.input,                               outputs=model.get_layer(name).output).predict(X_test)
            print(name+'\n', out)
    '''

    if file_output:
        with open(save_path+data_name+'_result.txt', 'w') as f:
            f.write('loss=%.4f, acc=%.4f' % (loss, acc))


#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test', 'train_loop'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--model', help='model file name (when load model, mode=test)')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--model_kind', choices=['origin', 'plus_CAR', 'plus_KenLM', 'plus_both', 'replace_CAR', 'replace_KenLM', 'replace_both'], default='origin', help='model file kind')

    # ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()
    print(args.mode)
    epoch=args.epoch
    my_model_kind=args.model_kind

    # 1.語彙データ読み込み
    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)

    #Ngram couhnt集計
    train_ans=CLOTH_path+'CLOTH_train_ans.txt'
    clothNg=Ngram()
    clothNg.read_file_first(train_ans)

    if args.mode == 'all' or args.mode == 'train_loop':
        weights_matrix = get_weight_matrix(vocab)
    else:
        weights_matrix = np.zeros((vocab.n_words, EMB_DIM))

    #通常時
    if args.mode != 'train_loop':

        # 2.モデル定義
        model = build_model(vocab.n_words, EMB_DIM, HIDDEN_DIM, weights_matrix, my_model_kind)

        #学習時
        if args.mode == 'all' or args.mode == 'mini':
            train_cloze=CLOTH_path+'CLOTH_train_cloze.txt'
            train_choices=CLOTH_path+'CLOTH_train_choices.txt'
            train_ans=CLOTH_path+'CLOTH_train_ans.txt'

            valid_cloze=CLOTH_path+'CLOTH_valid_cloze.txt'
            valid_choices=CLOTH_path+'CLOTH_valid_choices.txt'
            valid_ans=CLOTH_path+'CLOTH_valid_ans.txt'

            print("Reading train/valid data...")
            train_X=readCloze(train_cloze)
            train_C=readChoices(train_choices)
            train_Y=readAns(train_ans)

            valid_X=readCloze(valid_cloze)
            valid_C=readChoices(valid_choices)
            valid_Y=readAns(valid_ans)

            if args.mode == 'mini':
                epoch=min(5, args.epoch)
                train_X=train_X[:300]
                train_C=train_C[:300]
                train_Y=train_Y[:300]

                valid_X=valid_X[:300]
                valid_C=valid_C[:300]
                valid_Y=valid_Y[:300]

            train_data = (train_X, train_C, train_Y)
            val_data = (valid_X, valid_C, valid_Y)

            #モデルとか結果とかを格納するディレクトリの作成
            save_path=save_path+args.mode+'_MPNet'
            if os.path.exists(save_path)==False:
                os.mkdir(save_path)
            save_path=save_path+'/'
            plot_model(model, to_file=save_path+'model_'+args.model_kind+'.png', show_shapes=True)
            #model.summary()

            # 3.学習
            model = trainIters(clothNg, vocab, model, train_data, val_data, my_model_kind, n_iters=epoch, saveModel=True)
            print('Train end')

        #すでにあるモデルでテスト時
        else:
            save_path=args.model_dir+'/'
            model.load_weights(save_path+args.model+'.hdf5')
            #model.summary()

            save_path=save_path+today_str

        # 4.評価
        center_cloze=git_data_path+'center_cloze.txt'
        center_choi=git_data_path+'center_choices.txt'
        center_ans=git_data_path+'center_ans.txt'

        MS_cloze=git_data_path+'microsoft_cloze.txt'
        MS_choi=git_data_path+'microsoft_choices.txt'
        MS_ans=git_data_path+'microsoft_ans.txt'

        high_path=git_data_path+'CLOTH_test_high'
        middle_path=git_data_path+'CLOTH_test_middle'

        CLOTH_high_cloze = high_path+'_cloze.txt'
        CLOTH_high_choi = high_path+'_choices.txt'
        CLOTH_high_ans = high_path+'_ans.txt'

        CLOTH_middle_cloze = middle_path+'_cloze.txt'
        CLOTH_middle_choi = middle_path+'_choices.txt'
        CLOTH_middle_ans = middle_path+'_ans.txt'

        is_out=False    #ファイル出力一括設定用

        #model_test(clothNg, vocab, model, center_cloze, center_choi, center_ans, my_model_kind, data_name='center', file_output=is_out)

        if args.mode != 'mini' and args.mode != 'mini_test':
            #model_test(clothNg, vocab, model, MS_cloze, MS_choi, MS_ans, data_name='MS', file_output=is_out)

            #model_test(clothNg, vocab, model, CLOTH_high_cloze, CLOTH_high_choi, CLOTH_high_ans, my_model_kind, data_name='CLOTH_high', file_output=is_out)

            model_test(clothNg, vocab, model, CLOTH_middle_cloze, CLOTH_middle_choi, CLOTH_middle_ans, my_model_kind, data_name='CLOTH_middle', file_output=is_out)

    #ループして複数モデル学習放置用
    else:
        train_cloze=CLOTH_path+'CLOTH_train_cloze.txt'
        train_choices=CLOTH_path+'CLOTH_train_choices.txt'
        train_ans=CLOTH_path+'CLOTH_train_ans.txt'

        valid_cloze=CLOTH_path+'CLOTH_valid_cloze.txt'
        valid_choices=CLOTH_path+'CLOTH_valid_choices.txt'
        valid_ans=CLOTH_path+'CLOTH_valid_ans.txt'

        print("Reading train/valid data...")
        train_X=readCloze(train_cloze)
        train_C=readChoices(train_choices)
        train_Y=readAns(train_ans)

        valid_X=readCloze(valid_cloze)
        valid_C=readChoices(valid_choices)
        valid_Y=readAns(valid_ans)

        train_data = (train_X, train_C, train_Y)
        val_data = (valid_X, valid_C, valid_Y)

        center_cloze=git_data_path+'center_cloze.txt'
        center_choi=git_data_path+'center_choices.txt'
        center_ans=git_data_path+'center_ans.txt'

        MS_cloze=git_data_path+'microsoft_cloze.txt'
        MS_choi=git_data_path+'microsoft_choices.txt'
        MS_ans=git_data_path+'microsoft_ans.txt'

        high_path=git_data_path+'CLOTH_test_high'
        middle_path=git_data_path+'CLOTH_test_middle'

        CLOTH_high_cloze = high_path+'_cloze.txt'
        CLOTH_high_choi = high_path+'_choices.txt'
        CLOTH_high_ans = high_path+'_ans.txt'

        CLOTH_middle_cloze = middle_path+'_cloze.txt'
        CLOTH_middle_choi = middle_path+'_choices.txt'
        CLOTH_middle_ans = middle_path+'_ans.txt'

        is_out=True    #ファイル出力一括設定用

        # 2.モデル定義
        models=['plus_CAR', 'replace_CAR', 'plus_KenLM', 'plus_both' , 'replace_KenLM', 'replace_both']

        #済
        #'origin', 

        for my_model_kind in models:

            start_date=datetime.datetime.today()
            start_date_str=today1.strftime('%m_%d_%H%M')
            save_path=file_path + start_date_str

            model = build_model(vocab.n_words, EMB_DIM, HIDDEN_DIM, weights_matrix, my_model_kind)

            #モデルとか結果とかを格納するディレクトリの作成
            save_path=save_path+'_MPNet_'+my_model_kind
            if os.path.exists(save_path)==False:
                os.mkdir(save_path)
            save_path=save_path+'/'
            plot_model(model, to_file=save_path+'model_'+args.model_kind+'.png', show_shapes=True)
            #model.summary()

            # 3.学習
            model = trainIters(clothNg, vocab, model, train_data, val_data, my_model_kind, n_iters=epoch, saveModel=True)
            print('Train end')

            # 4.評価
            model_test(clothNg, vocab, model, center_cloze, center_choi, center_ans, my_model_kind, data_name='center', file_output=is_out)


            #model_test(clothNg, vocab, model, MS_cloze, MS_choi, MS_ans, data_name='MS', file_output=is_out)

            model_test(clothNg, vocab, model, CLOTH_high_cloze, CLOTH_high_choi, CLOTH_high_ans, my_model_kind, data_name='CLOTH_high', file_output=is_out)

            model_test(clothNg, vocab, model, CLOTH_middle_cloze, CLOTH_middle_choi, CLOTH_middle_ans, my_model_kind, data_name='CLOTH_middle', file_output=is_out)

            del model
            #一応メモリ解放
