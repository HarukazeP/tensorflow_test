# -*- coding: utf-8 -*-

'''

※ 未完成

※ embedding層の返り値がnanになるためうまく動作しない
※ 原因は不明のまま

Liang Wangらnoの論文
「Multi-Perspective Context Aggregation for Semi-supervised Cloze-style Reading Comprehension」
ttps://aclanthology.coli.uni-saarland.de/papers/C18-1073/c18-1073
を読んでモデル自分なりにできる範囲で真似したやつ

論文のモデル厳密には再現できていない
半教師あり学習は未実装

----- 論文に書いてたこと -----
!! は再現できていないところ

!! ハイパーパラメータは検証データでのランダムサーチにより決定
!! embeddingの初期値はGloveの300次元，上位1000単語のみ学習で更新   #これってもしかして1000語のEmbeddingしてる？
!! 最適化関数はadam
!! 学習率は最初10^(-3)，学習を重ねるごとに10^(-4)や10^(-5)に小さくしていく
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，フィルター数128，width3
GRUのドロップアウト率は50%


----- 自分の実装 -----
#TODO これ書いてるだけ

!! はもとの論文と異なる点
それ以外にも，次元数というか階数が元の論文あまり書いてなかったから割と違うかも

!! 語彙数は上位3万単語
!! embeddingの初期値はGoogleの学習済みword2vecの300次元，全て学習で更新
!! 最適化関数はSGD
!! 学習率は最初10^(-3)，val_lossが減少しない場合は学習率小さくする
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，出力次元数128，kernel_size 3
!! CNNのバッチ正規化とかReLUの利用場所も曖昧
GRUのドロップアウト率は50%
次元数というか階数が元の論文あまり書いてなかったから割と違うかも

動かしていたバージョン
python  : 3.5.2 / 3.6.5
pytorch : 2.0.4
gensim  : 3.1.0 / 3.5.0


'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import datetime

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse
import collections

import copy

from torch.utils.data import TensorDataset, DataLoader

import gensim
import nltk

#----- グローバル変数一覧 -----
MAX_LENGTH = 80
C_MAXLEN = 6
HIDDEN_DIM = 128
EMB_DIM = 300
BATCH_SIZE = 256

file_path='../../../pytorch_data/'
git_data_path='../../Data/'
CLOTH_path = file_path+'CLOTH_for_model/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str
PAD_token = 0
UNK_token = 1
CLZ_token = 2

CLZ_word = 'XXXX'

#事前処理いろいろ
print('Start: '+today_str)
if torch.cuda.is_available():
    my_device = torch.device("cuda")
    print('Use GPU')
else:
    my_device= torch.device("cpu")
#my_CPU=torch.device("cpu")



#----- 関数群 -----

###########################
# 1.データの準備，データ変換
###########################

#語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": UNK_token}
        self.index2word = {PAD_token: "PAD", UNK_token: "<UNK>", CLZ_token: "CLZ"}
        self.n_words = 3  # PAD と UNK とCLZ

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


#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


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
            ids.append(lang.check_word2index('NUM'))
        else:
            ids.append(lang.check_word2index(word.lower()))

    return ids + [PAD_token] * (maxlen - len(ids))


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


#与えた語彙読み込み
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(line.strip())
    #print("Vocab: %s" % lang.n_words)

    return lang


#入出力データ読み込み用
def readData(input_file, target_file):
    #print("Reading data...")
    pairs=[]
    i=0
    with open(input_file, encoding='utf-8') as input:
        with open(target_file, encoding='utf-8') as target:
            for line1, line2 in zip(input, target):
                i+=1
                pairs.append([line1.strip(), line2.strip()])
    print("data: %s" % i)

    return pairs


#ペアじゃなくて単独で読み取るやつ
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


def readAns(file_name):
    data=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            data.append(line.strip())

    return data

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

# --- MPALayerの一部: Iterative Dilated Convolution ---
class IDCNN(nn.Module):
    def __init__(self, hid_dim, num_directions):
        super(IDCNN, self).__init__()
        self.hidden_dim = hid_dim
        #conv_in_dim=self.hidden_dim*num_directions
        conv_in_dim=MAX_LENGTH
        conv_out_dim=self.hidden_dim*num_directions

        self.CNN1_1=nn.Conv1d(conv_in_dim, conv_out_dim, kernel_size=3, dilation=1)
        self.CNN1_3=nn.Conv1d(conv_out_dim, conv_out_dim, kernel_size=3, dilation=3)
        self.CNN2_1=nn.Conv1d(conv_out_dim, conv_out_dim, kernel_size=3, dilation=1)
        self.CNN2_3=nn.Conv1d(conv_out_dim, conv_out_dim, kernel_size=3, dilation=3)

        self.bn1 = nn.BatchNorm1d(conv_out_dim)
        self.bn2 = nn.BatchNorm1d(conv_out_dim)

        self.MaxP=nn.AdaptiveMaxPool1d(output_size=1)
        #通常のMaxPooingはフィルタの大きさを指定して，フィルタ内の最大を返すが，
        #AdaptiveMaxPoolは出力次元を指定できる

    def forward(self, sents_vec):
        '''
            入力: sents_vec  (b, s, 2h)
            出力：cnn_vec (b, 2h)
        '''
        #TODO バッチ正規化とかreruとか，この位置だけでいい？

        x=F.relu(self.bn1(self.CNN1_1(sents_vec)))
        x=self.CNN1_3(x)

        x=F.relu(self.bn2(self.CNN2_1(x)))
        x=self.CNN2_3(x)

        x=self.MaxP(x)
        cnn_vec=x.squeeze()    # (b, 2h, 1) -> (b, 2h)

        return cnn_vec #(b, 2h)


# --- MPALayerの一部: Attentive Reader ---
class AttnReader(nn.Module):
    def __init__(self, hid_dim, num_directions):
        super(AttnReader, self).__init__()
        self.hidden_dim = hid_dim

        self.weight = Parameter(torch.Tensor(self.hidden_dim*num_directions, self.hidden_dim*num_directions))
        self.bias = Parameter(torch.Tensor(self.hidden_dim*num_directions))

    def forward(self, sents_vec, c1_vec, c2_vec, c3_vec, c4_vec):
        '''
            入力: sents_vec  (b, s, 2h)
                  c1_vecなど  (b, 2h)
            出力：p1など      (b, 2h)
        '''

        Wh=torch.matmul(sents_vec, self.weight) # (b, s, 2h)

        bh=torch.matmul(sents_vec, self.bias) # (b, s)
        bh=bh.unsqueeze(2)  # (b, s, 1)

        u1=c1_vec.unsqueeze(2) # (b, 2h) -> (b, 2h, 1)
        u2=c2_vec.unsqueeze(2)
        u3=c3_vec.unsqueeze(2)
        u4=c4_vec.unsqueeze(2)

        u1_Wh=torch.bmm(Wh, u1) # (b, s, 1)
        u2_Wh=torch.bmm(Wh, u2)
        u3_Wh=torch.bmm(Wh, u3)
        u4_Wh=torch.bmm(Wh, u4)

        attn_1=F.softmax(u1_Wh+bh, dim=1) # (b, s, 1)
        attn_2=F.softmax(u2_Wh+bh, dim=1)
        attn_3=F.softmax(u3_Wh+bh, dim=1)
        attn_4=F.softmax(u4_Wh+bh, dim=1)

        attn1_h=torch.mul(sents_vec, attn_1)  # (b, s, 2h)
        attn2_h=torch.mul(sents_vec, attn_2)
        attn3_h=torch.mul(sents_vec, attn_3)
        attn4_h=torch.mul(sents_vec, attn_4)

        P1=torch.sum(attn1_h, dim=1)    # (b, s, 2h) -> (b, 2h)
        P2=torch.sum(attn2_h, dim=1)
        P3=torch.sum(attn3_h, dim=1)
        P4=torch.sum(attn4_h, dim=1)

        return P1, P2, P3, P4


# --- 論文中のMulti-Perspective Aggregation Layer ---
class MPALayer(nn.Module):
    def __init__(self, hid_dim, out_dim, num_directions):
        super(MPALayer, self).__init__()
        self.hidden_dim = hid_dim
        self.output_dim = out_dim   #選択肢の数

        #self.SC=
        self.IDC=IDCNN(self.hidden_dim, num_directions)
        self.AR=AttnReader(self.hidden_dim, num_directions)
        #self.NG=


    def forward(self, sents_vec, input_batch, c1_vec, c2_vec, c3_vec, c4_vec):
        '''
            入力: sents_vec    (b, s, 2h)
                  input_batch  (b, s)
                  c1_vec       (b, 2h)

            出力：PとC1,C2,C3,C4
                P  : (b, 4h)
                C1 : (b, 6h)
        '''
        # --- Selective Copying ---


        # --- Iterative Dilated Convolution ---
        P_idc=self.IDC(sents_vec)   #(b,2h)


        # --- Attentive Reader ---
        P1_ar, P2_ar, P3_ar, P4_ar=self.AR(sents_vec, c1_vec, c2_vec, c3_vec, c4_vec)
        # P1_ar: (b, 2h)


        # --- N-gram Statistics ---
        '''
            入力：input_batch  (s, b)
            出力：ngram_prob (？, b, ？？？) #TODO これ次元数どうなる
        '''


        #最後にマージ
        '''
        PはP_scとP_idcの連結 (b, (h+h)*num_directions)？
        C1はc1_vecとP1_arとP1_ngの連結 (b, (h+h+h)*num_directions)？

        P  : (b, 2*num_directions*h)
        C1 : (b, 3*num_directions*h)
        '''
        P=P_idc

        C1=torch.cat([c1_vec, P1_ar], dim=1)
        C2=torch.cat([c2_vec, P2_ar], dim=1)
        C3=torch.cat([c3_vec, P3_ar], dim=1)
        C4=torch.cat([c4_vec, P4_ar], dim=1)

        return P, C1, C2, C3, C4


class PointerNet(nn.Module):
    def __init__(self, hid_dim, out_dim, num_directions):
        super(PointerNet, self).__init__()
        self.hidden_dim = hid_dim
        self.output_dim = out_dim   #選択肢の数
        #TODO ここ変更した場所
        '''
        self.P_dim=2*num_directions
        self.C_dim=3*num_directions
        '''
        self.P_dim=1*num_directions
        self.C_dim=2*num_directions

        self.GateWeight_P = Parameter(torch.Tensor(self.hidden_dim*self.C_dim, self.hidden_dim*self.P_dim))
        self.GateWeight_C = Parameter(torch.Tensor(self.hidden_dim*self.C_dim, self.hidden_dim*self.C_dim))
        self.GateBias = Parameter(torch.Tensor(self.hidden_dim*self.C_dim))

        self.OutWeight_P = Parameter(torch.Tensor(self.hidden_dim*self.C_dim, self.hidden_dim*self.P_dim))
        self.OutBias = Parameter(torch.Tensor(self.hidden_dim*self.C_dim))

    def forward(self, P, C1, C2, C3, C4, batch_size):
        '''
            入力:PとC1,C2,C3,C4
            出力： (b, n)  #各選択肢に対する確率

            P  : (b, 2h)
            C1 : (b, 4h)

            matmul()やbmm()は内積，mul()は要素積

            print('', .size())
        '''
        WP=P.matmul(self.GateWeight_P.t())      # (b, 2h) -> (b, 4h)
        WC1=C1.matmul(self.GateWeight_C.t())    # (b, 4h) -> (b, 4h)
        WC2=C2.matmul(self.GateWeight_C.t())
        WC3=C3.matmul(self.GateWeight_C.t())
        WC4=C4.matmul(self.GateWeight_C.t())

        g1=WP+WC1+self.GateBias #(b, 4h)　#これでちゃんとバッチ数分バイス足せてる
        g2=WP+WC2+self.GateBias #(b, 4h)
        g3=WP+WC3+self.GateBias #(b, 4h)
        g4=WP+WC4+self.GateBias #(b, 4h)

        C_dash1=torch.mul(C1, torch.sigmoid(g1))    #(b, 4h)
        C_dash2=torch.mul(C2, torch.sigmoid(g2))
        C_dash3=torch.mul(C3, torch.sigmoid(g3))
        C_dash4=torch.mul(C4, torch.sigmoid(g4))

        WP_out=P.matmul(self.OutWeight_P.t())   # (b, 2h) -> (b, 4h)

        #(b, 6h)と(b,6h)の内積で(b,1)にしたいが
        #pytorchでバッチごとの内積はbmmしかないため変形してる
        C1WP=torch.bmm(C_dash1.view(batch_size, 1, self.hidden_dim*self.C_dim), WP_out.view(batch_size, self.hidden_dim*self.C_dim, 1))   #(b,1,1)
        C1WP=C1WP.squeeze(2) #(b,1)
        C2WP=torch.bmm(C_dash2.view(batch_size, 1, self.hidden_dim*self.C_dim), WP_out.view(batch_size, self.hidden_dim*self.C_dim, 1))
        C2WP=C2WP.squeeze(2)
        C3WP=torch.bmm(C_dash3.view(batch_size, 1, self.hidden_dim*self.C_dim), WP_out.view(batch_size, self.hidden_dim*self.C_dim, 1))
        C3WP=C3WP.squeeze(2)
        C4WP=torch.bmm(C_dash4.view(batch_size, 1, self.hidden_dim*self.C_dim), WP_out.view(batch_size, self.hidden_dim*self.C_dim, 1))
        C4WP=C4WP.squeeze(2)

        bC1=torch.matmul(C_dash1, self.OutBias) #(b)
        bC1=bC1.unsqueeze(1)    #(b,1)
        bC2=torch.matmul(C_dash2, self.OutBias)
        bC2=bC2.unsqueeze(1)
        bC3=torch.matmul(C_dash3, self.OutBias)
        bC3=bC3.unsqueeze(1)
        bC4=torch.matmul(C_dash4, self.OutBias)
        bC4=bC4.unsqueeze(1)

        out1=C1WP+bC1   #(b,1)
        out2=C1WP+bC1
        out3=C1WP+bC1
        out4=C1WP+bC1

        output=torch.cat([out1, out2, out3, out4], dim=1)   #(b,4)
        #output=F.softmax(output, dim=1)   #(b,4)
        #output = F.log_softmax(output, dim=1)

        return output #(b, 4)


#できる限り再現したモデル
#選択肢は4つ限定
class MPnet(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, out_dim, weights_matrix, BiDi=True):
        super(MPnet, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = emb_dim
        self.hidden_dim = hid_dim
        self.output_dim = out_dim   #選択肢の数
        num_directions = 2 if BiDi else 1

        # --- 論文中のInput layer ---
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=PAD_token) #語彙数×次元数
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

        self.BiGRU=nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=BiDi)

        self.choicesBiGRU=nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, num_layers=1, batch_first=True, bidirectional=BiDi)
        self.choicesLiner = nn.Linear(self.hidden_dim*num_directions*C_MAXLEN, self.hidden_dim*num_directions)

        # --- 論文中のMulti-Perspective Aggregation Layer ---
        self.MPA=MPALayer(self.hidden_dim, self.output_dim, num_directions)

        # --- 論文中のOutput Layer ---
        self.OutLayer=PointerNet(self.hidden_dim, self.output_dim, num_directions)
    '''
    def make_choice_vec(choi):
        new_choi=choi.squeeze()   # (b, 1, c) -> (b, c)
        c_emb = self.embedding(new_choi) # (b, c) -> (b, c, h)
        c_gru, _ = self.choicesBiGRU(c_emb)    # (b, c, h) -> (b, c, 2h) , (b, 1*2, h)

        c_out=c_gru.contiguous().view(batch_size, -1) # (b, c*2h)
        c_vec=self.choicesLiner(c_out)    # (b, 2h)

        return c_vec
    '''

    def forward(self, input_batch, choices_batch):
        """
        :param
            input_batch:   (b, s)
            choices_batch: (b, n, c)
                n:1問あたりの選択肢数
                c:選択肢の最大長(パディング後)


        :returns (b, 4)
        """
        # --- 論文中のInput layer ---
        #TODO inputからCLZの場所特定しておく？→それバッチでできる？
        batch_size = input_batch.shape[0]
        embedded = self.embedding(input_batch)  # (b, s) -> (b, s, h)
        sent_gru, _ = self.BiGRU(embedded)    # (b, s, h) -> (b, s, 2h) , (b, 1*2, h)
        sent_vec = F.dropout(sent_gru, p=0.5, training=self.training)

        c1, c2, c3, c4 =torch.chunk(choices_batch, self.output_dim, dim=1)

        '''
        こう書きたかったけど何故かエラー出る
        c1_vec = self.make_choice_vec(c1)
        TypeError: make_choice_vec() takes 1 positional argument but 2 were given

        c1_vec = self.make_choice_vec(c1)
        c2_vec = self.make_choice_vec(c2)
        c3_vec = self.make_choice_vec(c3)
        c4_vec = self.make_choice_vec(c4)
        '''
        # c1_vec作成
        c1=c1.squeeze()   # (b, 1, c) -> (b, c)
        c1_gru, _ = self.choicesBiGRU(self.embedding(c1))
        c1_gru = F.dropout(c1_gru, p=0.5, training=self.training)
        c1_out=c1_gru.contiguous().view(batch_size, -1) # (b, c*2h)
        c1_vec=self.choicesLiner(c1_out)    # (b, 2h)

        # c2_vec作成
        c2=c2.squeeze()
        c2_gru, _ = self.choicesBiGRU(self.embedding(c2))
        c2_gru = F.dropout(c2_gru, p=0.5, training=self.training)
        c2_out=c2_gru.contiguous().view(batch_size, -1)
        c2_vec=self.choicesLiner(c2_out)

        # c3_vec作成
        c3=c3.squeeze()
        c3_gru, _ = self.choicesBiGRU(self.embedding(c3))
        c3_gru = F.dropout(c3_gru, p=0.5, training=self.training)
        c3_out=c3_gru.contiguous().view(batch_size, -1)
        c3_vec=self.choicesLiner(c3_out)

        # c4_vec作成
        c4=c4.squeeze()
        c4_gru, _ = self.choicesBiGRU(self.embedding(c4))
        c4_gru = F.dropout(c4_gru, p=0.5, training=self.training)
        c4_out=c4_gru.contiguous().view(batch_size, -1)
        c4_vec=self.choicesLiner(c4_out)


        # --- 論文中のMulti-Perspective Aggregation Layer ---
        P, C1, C2, C3, C4=self.MPA(sent_vec, input_batch, c1_vec, c2_vec, c3_vec, c4_vec)
        # P: (b, 2h),  C:(b, 4h)


        # --- 論文中のOutput Layer ---
        output=self.OutLayer(P, C1, C2, C3, C4, batch_size) # (b, 4)

        #デバッグ用
        '''
        print('sent',input_batch[5])
        print('sent emb',embedded[5])
        #print('choices',choices_batch[5])
        #print('c1',c1[5])
        #print('sent_gru:',sent_gru[5])
        #print('c1_gru:',c1_gru[5])
        #print('sent_vec:',sent_vec[5])
        #print('c1_vec:',c1_vec[5])
        #print('P:',P[5])
        #print('C1:',C1[5])
        #print('model output:',output[5])
        '''

        return output   # (b, 4)






###########################
# 3.モデルの学習
###########################

#バッチデータあたりの学習
def batch_train(X, C, Y, model, model_optimizer, criterion, max_length=MAX_LENGTH):
    '''
    X : (s, b)
    X : (s, b)
    Y : (b, 4)
    '''
    #loss=0
    model_optimizer.zero_grad()

    model_outputs = model(X, C) #出力 (b, 4)

    #pytorchでNLLlossのpredは(b,4)だけどtargetは(4)
    loss = criterion(model_outputs, torch.argmax(Y, dim=1))

    loss.backward()
    model_optimizer.step()  #重みの更新

    return loss.item()



#バッチデータあたりのバリデーション
def batch_eval(X, C, Y, model, criterion):
    with torch.no_grad():
        loss=0

        model_outputs = model(X, C) #出力 (b, 4)
        loss = criterion(model_outputs, torch.argmax(Y, dim=1))
        acc = calc_batch_acc(model_outputs, Y)

        return loss.item(), acc

def calc_batch_acc(model_outputs, Y):
    batch=0
    OK=0
    model_pred=torch.argmax(model_outputs, dim=1)
    y=torch.argmax(Y, dim=1)
    for pred, ans in zip(model_pred, y):
        batch+=1
        if pred==ans:
            OK+=1

    acc = 100.0*OK/batch
    #print('OK: %d batch: %d acc: %.2f %%' % (OK, batch, acc))

    return acc

#秒を分秒に変換
def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


#経過時間と残り時間の算出
def timeSince(since, percent):
    now = time.time()
    s = now - since       #経過時間
    es = s / (percent)    #終了までにかかる総時間
    rs = es - s           #終了までの残り時間
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


#学習をn_iters回，残り時間の算出をlossグラフの描画も
def trainIters(lang, model, train_pairs, val_pairs, n_iters, print_every=10, learning_rate=0.001, saveModel=False):
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0

    plot_val_losses = []
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0

    plot_val_accs = []
    print_val_acc_total = 0  # Reset every print_every
    plot_val_acc_total = 0

    best_iter=0
    best_val_acc = None

    best_model_weight = copy.deepcopy(model.state_dict())

    model_optimizer = optim.SGD(model.parameters(), lr=learning_rate)
    X_train=[sent_to_ids_cloze(lang, s) for s in train_pairs[0]]
    C_train=[choices_to_ids(lang, s) for s in train_pairs[1]]
    Y_train=[ans_to_ids(lang, s, c) for s,c in zip(train_pairs[2], train_pairs[1])]

    X_val=[sent_to_ids_cloze(lang, s) for s in val_pairs[0]]
    C_val=[choices_to_ids(lang, s) for s in val_pairs[1]]
    Y_val=[ans_to_ids(lang, s, c) for s,c in zip(val_pairs[2], val_pairs[1])]

    train_data_num=len(X_train)
    val_data_num=len(X_val)
    print('train data:', train_data_num)
    print('valid data:', val_data_num)

    '''
    #データ全体はRAMに載せる
    X_train=torch.tensor(X_train, dtype=torch.long, device=my_CPU)
    C_train=torch.tensor(C_train, dtype=torch.long, device=my_CPU)
    Y_train=torch.tensor(Y_train, dtype=torch.long, device=my_CPU)
    X_val=torch.tensor(X_val, dtype=torch.long, device=my_CPU)
    C_val=torch.tensor(C_val, dtype=torch.long, device=my_CPU)
    Y_val=torch.tensor(Y_val, dtype=torch.long, device=my_CPU)
    '''
    X_train=torch.tensor(X_train, dtype=torch.long, device=my_device)
    C_train=torch.tensor(C_train, dtype=torch.long, device=my_device)
    Y_train=torch.tensor(Y_train, dtype=torch.long, device=my_device)
    X_val=torch.tensor(X_val, dtype=torch.long, device=my_device)
    C_val=torch.tensor(C_val, dtype=torch.long, device=my_device)
    Y_val=torch.tensor(Y_val, dtype=torch.long, device=my_device)

    ds_train = TensorDataset(X_train, C_train, Y_train)
    ds_val = TensorDataset(X_val, C_val, Y_val)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)

    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()

    start = time.time()
    st_time=datetime.datetime.today().strftime('%H:%M')
    print("Training... ", st_time)

    # Ctrl+c で強制終了してもそこまでのモデルとか保存
    try:
        for iter in range(1, n_iters + 1):
            for x, c, y in loader_train:
                '''
                x:(バッチサイズ, 文長)
                y:(バッチサイズ, 文長)
                '''
                '''
                #計算はGPUのメモリ上で
                x=x.to(my_device)
                c=c.to(my_device)
                y=y.to(my_device)
                '''
                batch=x.size(0)

                loss= batch_train(x, c, y, model, model_optimizer, criterion)
                #NLLLossはバッチ平均を返すから元に戻す
                loss=loss*batch

                print_loss_total += loss
                plot_loss_total += loss
            #ここで学習1回分終わり

            for x, c, y in loader_val:
                '''
                x=x.to(my_device)
                c=c.to(my_device)
                y=y.to(my_device)
                '''
                batch=x.size(0)

                val_loss, val_acc = batch_eval(x, c, y, model, criterion)
                val_loss=val_loss*batch
                val_acc=val_acc*batch

                print_val_loss_total += val_loss
                plot_val_loss_total += val_loss

                print_val_acc_total += val_acc
                plot_val_acc_total += val_acc

            #画面にlossと時間表示
            #経過時間 (- 残り時間) (現在のiter 進行度) loss val_loss
            if iter == 1:
                print('%s (%d %d%%) train_loss=%.4f, val_loss=%.4f, val_acc=%.2f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_total/train_data_num, print_val_loss_total/val_data_num, print_val_acc_total/val_data_num))

            elif iter % print_every == 0:
                print_loss_avg = (print_loss_total/train_data_num) / print_every
                print_loss_total = 0
                print_val_loss_avg = (print_val_loss_total/val_data_num) / print_every
                print_val_loss_total = 0
                print_val_acc_avg = (print_val_acc_total/val_data_num) / print_every
                print_val_acc_total = 0
                print('%s (%d %d%%) train_loss=%.4f, val_loss=%.4f, val_acc=%.2f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, print_val_loss_avg, print_val_acc_avg))

            #loss/accグラフ用記録
            plot_loss_avg = plot_loss_total/train_data_num
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

            plot_val_loss_avg = plot_val_loss_total/val_data_num
            plot_val_losses.append(plot_val_loss_avg)
            plot_val_loss_total = 0

            #TODO acc はaveとかいる？
            plot_val_acc_avg = plot_val_acc_total/val_data_num
            plot_val_accs.append(plot_val_acc_avg)
            plot_val_acc_total = 0

            #val_loss最小更新
            if not best_val_acc or val_acc > best_val_acc:
                best_val_acc = val_acc
                best_iter=iter
                best_model_weight = copy.deepcopy(model.state_dict())
            else:
                # val_accが向上しない場合は学習率小さくする
                learning_rate /= 4.0    #この4に特に意味はない，マジックナンバー

    except KeyboardInterrupt:
        print('-' * 89)
        if best_iter >=0:
            print('Exiting from training early')
        else :
            exit()

    #全学習終わり
    #lossグラフ描画
    showPlot3(plot_losses, plot_val_losses, 'loss.png', 'loss')
    showPlot4(plot_val_accs, 'acc.png', 'acc')
    #showPlot2(plot_accs, plot_val_accs, 'acc.png')

    #val_loss最小のモデルロード
    model.load_state_dict(best_model_weight)
    print('best iter='+str(best_iter))

    if saveModel:
        torch.save(model.state_dict(), save_path+'MPnet_'+str(best_iter)+'.pth')

    return model


#グラフの描画（画像ファイル保存）
def showPlot3(train_plot, val_plot, file_name, label_name):
    #fig = plt.figure()
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
'''
#TODO テスト部分後日実装

    # 1データに対する予測
    def evaluate(lang, model, sentence, max_length=MAX_LENGTH):
        with torch.no_grad():
            #no_grad()の間はパラメータが固定される（更新されない）
            input_indexes = pad_indexes(lang, sentence)
            input_batch = torch.tensor([input_indexes], dtype=torch.long, device=my_device)  # (1, s)

            encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1))

            decoder_input = torch.tensor([SOS_token], device=my_device)  # SOS
            decoder_hidden = (
                (encoder_hidden[0][0].squeeze(0), encoder_hidden[1][0].squeeze(0)),
                (encoder_hidden[0][1].squeeze(0), encoder_hidden[1][1].squeeze(0))
                )

            decoded_words = []
            decoder_attentions = []

            for di in range(max_length):
                decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)
                decoder_attentions.append(attention)
                _, topi = decoder_output.topk(1)  # (1, 1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(lang.index2word[topi.item()])

                decoder_input = topi[0]

            decoder_attentions = torch.cat(decoder_attentions, dim=0)  # (l, n)

            #返り値は予測した単語列とattentionの重み？
            return decoded_words, decoder_attentions.squeeze(0)


    #attentionの重みの対応グラフの描画
    def showAttention(file_header, input_sentence, output_words, attentions):
        #TODO 描画方法は要改善
        #目盛り間隔、軸ラベルの位置など

        fig = plt.figure()
        ax = fig.add_subplot(111)
        cax = ax.matshow(attentions.numpy().T, cmap='bone')
        fig.colorbar(cax)

        ax.set_yticklabels([''] + input_sentence.split(' ') +
                           ['<EOS>'])
        ax.set_xticklabels([''] + output_words, rotation=90)

        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        if len(input_sentence)>10:
            plt.savefig(save_path + file_header + input_sentence[:10] + '_attn.png')
        else:
            plt.savefig(save_path + file_header + input_sentence + '_attn.png')


    #精度いろいろ計算
    #問題文、完全一致文、空所の完答文、空所の一部正答文、BLEU値、空所ミス文
    def calc_score(preds_sentences, ans_sentences):
        line_num=0
        allOK=0
        clozeOK=0
        partOK=0
        miss=0
        BLEU=0

        for pred, ans in zip(preds_sentences, ans_sentences):
            pred=pred.replace(' <EOS>', '')
            flag=0
            if pred == ans:
                allOK+=1
                flag=1
            pred_cloze = get_cloze(pred)
            ans_cloze = get_cloze(ans)
            tmp_ans_length=len(ans_cloze.split(' '))
            line_num+=1
            if is_correct_cloze(pred):
                tmp_match=match(pred_cloze, ans_cloze)
                if tmp_match > 0:
                    partOK+=1
                if pred_cloze == ans_cloze:
                    clozeOK+=1
                    if flag==0:
                        print(pred)
                        print(ans)
            else:
                miss+=1

        #BLEU=compute_bleu(preds_sentences, ans_sentences)

        return line_num, allOK, clozeOK, partOK, BLEU, miss


    def output_preds(file_name, preds):
        with open(file_name, 'w') as f:
            for p in preds:
                f.write(p+'\n')


    def print_score(line, allOK, clozeOK, partOK, BLEU, miss):
        print('  acc(all): ', '{0:.2f}'.format(1.0*allOK/line*100),' %')
        #print('acc(cloze): ', '{0:.2f}'.format(1.0*clozeOK/line*100),' %')
        #print(' acc(part): ', '{0:.2f}'.format(1.0*partOK/line*100),' %')

        #print(' BLEU: ','{0:.2f}'.format(BLEU*100.0))
        print('  all: ', allOK)
        #print('cloze: ',clozeOK)
        #print(' part: ',partOK)
        print(' line: ',line)
        print(' miss: ',miss)


    def output_score(file_name, line, allOK, clozeOK, partOK, BLEU, miss):
        output=''
        output=output+'  acc(all): '+str(1.0*allOK/line*100)+' %\n'
        #output=output+'acc(cloze): '+str(1.0*clozeOK/line*100)+' %\n'
        #output=output+' acc(part): '+str(1.0*partOK/line*100)+' %\n\n'
        #output=output+'      BLEU: '+str(BLEU*100.0)+' %\n\n'
        output=output+'       all: '+str(allOK)+'\n'
        #output=output+'     cloze: '+str(clozeOK)+'\n'
        #output=output+'      part: '+str(partOK)+'\n'
        output=output+'      line: '+str(line)+'\n'
        output=output+'      miss: '+str(miss)+'\n'

        with open(file_name, 'w') as f:
            f.write(output)


    def score(preds, ans, file_output, file_name):
        #精度のprintとファイル出力
        line, allOK, clozeOK, partOK, BLEU, miss = calc_score(preds, ans)
        print_score(line, allOK, clozeOK, partOK, BLEU, miss)
        if file_output:
            output_score(file_name, line, allOK, clozeOK, partOK, BLEU, miss)


    #テストデータに対する予測と精度計算
    #空所内のみを予測するモード
    #および、選択肢を利用するモード
    def test_choices(lang, model, test_data, choices, saveAttention=False, file_output=False):
        print("Test ...")
        #input_sentence や ansは文字列であるのに対し、output_wordsはリストであることに注意
        preds=[]
        ans=[]
        for pair, choi in zip(test_data, choices):
            input_sentence=pair[0]
            ans.append(pair[1])

            output_words, attentions = evaluate(lang, model, input_sentence)
            preds.append(' '.join(output_words))

            if saveAttention:
                showAttention('all', input_sentence, output_words, attentions)

            if file_output:
                output_preds(save_path+'preds.txt', preds)

        print("Calc scores ...")
        score(preds, ans, file_output, save_path+'score.txt')

'''

#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--model', help='model file name (when load model, mode=test)')
    parser.add_argument('--epoch', type=int, default=30)
    #TODO ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()
    print(args.mode)
    epoch=args.epoch

    # 1.語彙データ読み込み
    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)

    # 2.モデル定義
    if args.mode == 'all':
        weights_matrix = get_weight_matrix(vocab)
    else:
        weights_matrix = np.zeros((vocab.n_words, EMB_DIM))

    #デバッグ用
    '''
    print(torch.from_numpy(weights_matrix))
    exit()

    print('weights matrix')
    ct=0
    print(torch.from_numpy(weights matrix))
    for line in weights_matrix:
        for x in line:
            if math.isnan(x):
                print(line)
                break

    print('check end')
    exit()
    '''

    '''
    #マルチGPU対応
    if torch.cuda.device_count() > 1:
        print("Use", torch.cuda.device_count(), "GPUs")
        # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
        model = nn.DataParallel(MPnet(vocab.n_words, EMB_DIM, HIDDEN_DIM, 4, weights_matrix))
        model.to(my_device)
    else:
        model = MPnet(vocab.n_words, EMB_DIM, HIDDEN_DIM, 4, weights_matrix).to(my_device)
    '''
    model = MPnet(vocab.n_words, EMB_DIM, HIDDEN_DIM, 4, weights_matrix).to(my_device)

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
            epoch=5
            train_X=train_X[:300]
            train_C=train_C[:300]
            train_Y=train_Y[:300]

            valid_X=valid_X[:300]
            valid_C=valid_C[:300]
            valid_Y=valid_Y[:300]

        train_data = (train_X, train_C, train_Y)
        val_data = (valid_X, valid_C, valid_Y)

        #モデルとか結果とかを格納するディレクトリの作成
        save_path=save_path+args.mode+'_MPnet'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'

        # 3.学習
        model = trainIters(vocab, model, train_data, val_data, n_iters=epoch, saveModel=True)

    #すでにあるモデルでテスト時
    else:
        save_path=args.model_dir+'/'
        model.load_state_dict(torch.load(save_path+args.model))
        save_path=save_path+today_str

    # 4.評価
    print('Train end')

    '''
    #TODO テスト未実装

    center_cloze=git_data_path+'center_cloze.txt'
    center_ans=git_data_path+'center_ans.txt'
    center_choi=git_data_path+'center_choices.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'
    MS_choi=git_data_path+'microsoft_choices.txt'

    high_path=git_data_path+'CLOTH_test_high'
    middle_path=git_data_path+'CLOTH_test_middle'

    CLOTH_high_cloze = high_path+'_cloze.txt'
    CLOTH_high_ans = high_path+'_ans.txt'
    CLOTH_high_choi = high_path+'_choices.txt'

    CLOTH_middle_cloze = middle_path+'_cloze.txt'
    CLOTH_middle_ans = middle_path+'_ans.txt'
    CLOTH_middle_choi = middle_path+'_choices.txt'

    print("Reading test data...")
    center_X=readCloze(center_cloze)
    center_C=readChoices(center_choices)
    center_Y=readAns(center_ans)

    MS_X=readCloze(MS_cloze)
    MS_C=readChoices(MS_choices)
    MS_Y=readAns(MS_ans)

    CLOTH_high_X=readCloze(CLOTH_high_cloze)
    CLOTH_high_C=readChoices(CLOTH_high_choices)
    CLOTH_high_Y=readAns(CLOTH_high_ans)

    CLOTH_middle_X=readCloze(CLOTH_middle_cloze)
    CLOTH_middle_C=readChoices(CLOTH_middle_choices)
    CLOTH_middle_Y=readAns(CLOTH_middle_ans)

    if args.mode == 'mini' or args.mode == 'mini_test':
        center_data=center_data[:5]
        center_choices=center_choices[:5]
        MS_data=MS_data[:5]
        MS_choices=MS_choices[:5]


    #テストデータに対する予測と精度の計算
    #選択肢を使ったテスト
    #これは前からの予測
    print('center')
    tmp=save_path
    save_path=tmp+'center_'
    test_choices(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=True)

    #これは文スコア
    test_choices_by_sent_score(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=False)

    print('MS')
    tmp=save_path
    save_path=tmp+'MS_'
    test_choices(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=True)

    #これは文スコア
    test_choices_by_sent_score(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=False)

    '''
