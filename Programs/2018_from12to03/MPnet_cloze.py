# -*- coding: utf-8 -*-

'''
Liang Wangらnoの論文
「Multi-Perspective Context Aggregation for Semi-supervised Cloze-style Reading Comprehension」
ttps://aclanthology.coli.uni-saarland.de/papers/C18-1073/c18-1073
を読んでモデル自分なりにできる範囲で真似したやつ

論文のモデル厳密には再現できていない
半教師あり学習は未実装

----- 論文に書いてたこと -----
ハイパーパラメータは検証データでのランダムサーチにより決定
embeddingの初期値はGloveの300次元，上位1000単語のみ学習で更新
最適化関数はadam
学習率は最初10^(-3)，学習を重ねるごとに10^(-4)や10^(-5)に小さくしていく
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，フィルター数128，width3
GRUのドロップアウト率は50%


----- 自分の実装 -----
#TODO これ書いてるだけ

語彙数は上位3万単語
embeddingの初期値はGoogleの学習済みword2vecの300次元，全て学習で更新
最適化関数はSGD
学習率は最初10^(-3)，更新していく？
GRUは各128ユニット
入力は最大80単語
CNNでは2ブロック，出力次元数128，kernel_size 3
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
from sklearn.model_selection import train_test_split
import copy

from torch.utils.data import TensorDataset, DataLoader

import gensim

#----- グローバル変数一覧 -----
MAX_LENGTH = 40
C_MAXLEN = 6
HIDDEN_DIM = 128
ATTN_DIM = 128
EMB_DIM = 300
BATCH_SIZE = 128

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
git_data_path='../../Data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str
PAD_token = 0
UNK_token = 1
CLZ_token = 2

#事前処理いろいろ
print('Start: '+today_str)
if torch.cuda.is_available():
    my_device = torch.device("cuda")
    print('Use GPU')
else:
    my_device= torch.device("cpu")
my_CPU=torch.device("cpu")
#----- 関数群 -----


###########################
# 1.データの準備
###########################

#seq2seqモデルで用いる語彙に関するクラス
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


#データの前処理
#strip()は文頭文末の改行や空白を取り除いてくれる
def normalizeString(s, choices=False):
    s = unicodeToAscii(s.lower().strip())
    #text8コーパスと同等の前処理
    s=s.replace('0', ' zero ')
    s=s.replace('1', ' one ')
    s=s.replace('2', ' two ')
    s=s.replace('3', ' three ')
    s=s.replace('4', ' four ')
    s=s.replace('5', ' five ')
    s=s.replace('6', ' six ')
    s=s.replace('7', ' seven ')
    s=s.replace('8', ' eight ')
    s=s.replace('9', ' nine ')
    if choices:
        s = re.sub(r'[^a-z{}#]', ' ', s)
    else:
        s = re.sub(r'[^a-z{}]', ' ', s)
    s = re.sub(r'[ ]+', ' ', s)

    return s.strip()


#与えた語彙読み込み(自作)
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(normalizeString(line))
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
                pairs.append([normalizeString(line1), normalizeString(line2)])
    print("data: %s" % i)

    return pairs


#ペアじゃなくて単独で読み取るやつ
def readData2(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(normalizeString(line))

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
    def __init__(self, hid_dim, num_direction):
        super(PointerNet, self).__init__()
        self.hidden_dim = hid_dim
        conv_in_dim=self.hidden_dim*num_direction
        conv_out_dim=self.hidden_dim*num_direction

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
            入力: sents_vec  (s, b, 2h)
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
    def __init__(self, hid_dim, num_direction):
        super(AttnReader, self).__init__()
        self.hidden_dim = hid_dim
        self.output_dim = out_dim   #選択肢の数

        self.GateWeight_P = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim*P_dim))
        self.GateWeight_C = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim*C_dim))
        self.GateBias = Parameter(torch.Tensor(self.hidden_dim))

        self.weight = Parameter(torch.Tensor(self.output_dim*num_directions, self.output_dim*num_directions))
        self.bias = Parameter(torch.Tensor(self.output_dim*num_directions))

    def forward(self, sents_vec, c1_vec, c2_vec, c3_vec, c4_vec):
        '''
            入力: sents_vec  (s, b, 2h)
                  c1_vecなど  (b, 2h)
            出力：p1など      (b, 2h)
        '''

        Wh=torch.matmul(sents_vec, self.weight) # (b, s, h)

        bh=torch.matmul(sents_vec, self.bias) # (b, s)

        u1=c1_vec.unsqueeze(2) # (b, h) -> (b, h, 1)
        u2=c2_vec.unsqueeze(2)
        u3=c3_vec.unsqueeze(2)
        u4=c4_vec.unsqueeze(2)

        u1_Wh=torch.bmm(Wh, u1) # (b, s)
        u2_Wh=torch.bmm(Wh, u2)
        u3_Wh=torch.bmm(Wh, u3)
        u4_Wh=torch.bmm(Wh, u4)

        #これ計算あってる？適当にかいただけ
        attn1=u1_Wh+bh
        attn2=u2_Wh+bh
        attn3=u3_Wh+bh
        attn4=u4_Wh+bh


        #output=torch.cat([out1, out2, out3, out4], dim=1)   #(b,4)
        #prob=F.softmax(output, dim=1)   #(b,4)

        attn1_h=



        P1=torch.sum(attn1_h, dim=1)
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
        self.IDC=IDCNN(self.hidden_dim, num_direction)
        self.AR=AttnReader()
        #self.NG=


    def forward(self, sents_vec, input_batch, c1_vec, c2_vec, c3_vec, c4_vec):
        '''
            入力: sents_vec    (s, b, 2h)
                  input_batch  (s, b)
                  c1_vec       (b, 2h)

            出力：PとC1,C2,C3,C4
                P  : (b, 2h)
                C1 : (b, 3h)
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
        PはP_scとP_idcの連結 (b, (h+h*num_directions))？
        C1はc1_vecとP1_arとP1_ngの連結 (b, (h+h+h)*num_directions)？

        P  : (b, 2*num_directions*h)
        C1 : (b, 3*num_directions*h)
        '''

        return P, C1, C2, C3, C4


class PointerNet(nn.Module):
    # TODO: 引数とか適当
    def __init__(self, hid_dim, out_dim, num_directions):
        super(PointerNet, self).__init__()
        self.hidden_dim = hid_dim
        self.output_dim = out_dim   #選択肢の数
        P_dim=2*num_directions
        C_dim=3*num_directions

        self.GateWeight_P = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim*P_dim))
        self.GateWeight_C = Parameter(torch.Tensor(self.hidden_dim, self.hidden_dim*C_dim))
        self.GateBias = Parameter(torch.Tensor(self.hidden_dim))

        self.OutWeight_P = Parameter(torch.Tensor(self.output_dim, self.hidden_dim*P_dim))
        self.OutBias = Parameter(torch.Tensor(self.output_dim))

    def forward(self, P, C1, C2, C3, C4, batch_size):
        '''
            入力:PとC1,C2,C3,C4
            出力： (b, n)  #各選択肢に対する確率

            P  : (b, 2h)
            C1 : (b, 3h)

            matmul()やbmm()は内積，mul()は要素積
        '''

        WP=P.matmul(self.GateWeight_P.t())      # (b, 4h) -> (b, h)
        WC1=C1.matmul(self.GateWeight_C.t())    # (b, 6h) -> (b, h)
        WC2=C2.matmul(self.GateWeight_C.t())
        WC3=C3.matmul(self.GateWeight_C.t())
        WC4=C4.matmul(self.GateWeight_C.t())

        g1=WP+WC1+self.GateBias #(b, h)　#これでちゃんとバッチ数分バイス足せてる
        g2=WP+WC2+self.GateBias #(b, h)
        g3=WP+WC3+self.GateBias #(b, h)
        g4=WP+WC4+self.GateBias #(b, h)

        C_dash1=torch.mul(C1, F.sigmoid(g1))    #(b, h)
        C_dash2=torch.mul(C2, F.sigmoid(g2))
        C_dash3=torch.mul(C3, F.sigmoid(g3))
        C_dash4=torch.mul(C4, F.sigmoid(g4))

        WP_out=P.matmul(self.OutWeight_P.t())   # (b, 4h) -> (b, h)

        #(b, h)と(b,h)の内積で(b,1)にしたいが
        #pytorchでバッチごとの内積はbmmしかないため変形してる
        C1WP=torch.bmm(C_dash1.view(batch_size, 1, self.hidden_dim), WP_out.view(batch_size, self.hidden_dim, 1))   #(b,1,1)
        C1WP=C1WP.squeeze(2) #(b,1)
        C2WP=torch.bmm(C_dash2.view(batch_size, 1, self.hidden_dim), WP_out.view(batch_size, self.hidden_dim, 1))
        C2WP=C2WP.squeeze(2)
        C3WP=torch.bmm(C_dash3.view(batch_size, 1, self.hidden_dim), WP_out.view(batch_size, self.hidden_dim, 1))
        C3WP=C3WP.squeeze(2)
        C4WP=torch.bmm(C_dash4.view(batch_size, 1, self.hidden_dim), WP_out.view(batch_size, self.hidden_dim, 1))
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
        prob=F.softmax(output, dim=1)   #(b,4)

        return prob #(b, 4)


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

        self.BiGRU=nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=0.5, num_layers=1, bidirectional=BiDi)
        self.choicesBiGRU=nn.GRU(input_size=self.embedding_dim, hidden_size=self.hidden_dim, dropout=0.5, num_layers=1, bidirectional=BiDi)
        self.choicesLiner = nn.Linear(self.hidden_dim*num_directions*C_MAXLEN, self.hidden_dim*num_directions)

        # --- 論文中のMulti-Perspective Aggregation Layer ---
        self.MPA=MPALayer(self.hidden_dim, self.output_dim, num_directions)

        # --- 論文中のOutput Layer ---
        self.OutLayer=PointerNet(self.hidden_dim, self.output_dim, num_directions)

    def make_choices_vec(choice, Emb, GRU, Li):
        choice.squeeze(1)   # (c, 1, b) -> (c, b)
        c_emb = Emb(choice) # (c, b) -> (c, b, h)
        c_gru, _ = GRU(c_emb)    # (c, b, h) -> (c, b, 2h) , (1*2, b, h)
        c_out=c_gru.transpose(0,1)  # (c, b, 2h) -> (b, c, 2h)
        c_out=c_out.view(batch_size, -1) # (b, c*2h)
        c_vec=Li(c_out)    # (b, 2h)

        return c_vec

    def forward(self, input_batch, choices_batch):
        """
        :param
            input_batch:   (s, b)
            choices_batch: (c, n, b)
                c:選択肢の最大長(パディング後)
                n:1問あたりの選択肢数

        :returns (b, 4)
        """
        # --- 論文中のInput layer ---
        #TODO inputからCLZの場所特定しておく？→それバッチでできる？
        batch_size = input_batch.shape[1]

        embedded = self.embedding(input_batch)  # (s, b) -> (s, b, h)
        sent_vec, _ = self.BiGRU(embedded)    # (s, b, h) -> (s, b, h*2) , (1*2, b, h)

        c1, c2, c3, c4=torch.chunk(choices_batch, self.out_dim, dim=1)

        c1_vec = make_choices_vec(c1, self.embedding, self.choicesBiGRU, self.choicesLiner)
        c2_vec = make_choices_vec(c2, self.embedding, self.choicesBiGRU, self.choicesLiner)
        c3_vec = make_choices_vec(c3, self.embedding, self.choicesBiGRU, self.choicesLiner)
        c4_vec = make_choices_vec(c4, self.embedding, self.choicesBiGRU, self.choicesLiner)

        # --- 論文中のMulti-Perspective Aggregation Layer ---
        P, C1, C2, C3, C4=self.MPA(sent_vec, input_batch, c1_vec, c2_vec, c3_vec, c4_vec)
        # P: (b, 4h),  C:(b, 6h)

        # --- 論文中のOutput Layer ---
        output=self.OutLayer(P, C1, C2, C3, C4, batch_size) # (b, 4)

        return output   # (b, 4)



###########################
# 3.入力データ変換
###########################

#単語列をID列に
def indexesFromSentence(lang, sentence):
    return [lang.check_word2index(word) for word in sentence.split(' ')]


#単語列からモデルの入力へのテンソルに
#パディングあり、returnも変更
def pad_indexes(lang, sentence, dev=my_device):
    indexes = indexesFromSentence(lang, sentence, dev=dev)
    indexes.append(EOS_token)
    indexes + [PAD_token] * (MAX_LENGTH - len(indexes))
    return indexes + [PAD_token] * (MAX_LENGTH - len(indexes))





###########################
# 4.モデルの学習
###########################

'''
モデルの訓練

“Teacher forcing” は(seq2seqのでの)次の入力としてデコーダの推測を使用する代わりに、実際のターゲット出力を各次の入力として使用する概念です。

PyTorch autograd が与えてくれる自由度ゆえに、単純な if ステートメントで "teacher forcing" を使用するか否かをランダムに選択することができます。それを更に使用するためには teacher_forcing_ratio を上向きに調整してください。
'''

#1バッチデータあたりの学習
def batch_train(X, Y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    loss=0
    '''
    X : (s, b)
    Y : (s, b)
    '''

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    batch_size = X.size(1)
    target_length = Y.size(0)

    encoder_outputs, encoder_hidden = encoder(X) #出力 (s, b, 2h), ((1, b, h), (1, b, h))

    #デコーダの準備
    decoder_input = torch.tensor([[SOS_token] * batch_size], device=my_device)  # (1, b)
    decoder_inputs = torch.cat([decoder_input, Y], dim=0)  # (1,b), (n,b) -> (n+1, b)

    decoder_hidden = (
        (encoder_hidden[0][0].squeeze(0), encoder_hidden[1][0].squeeze(0)),
        (encoder_hidden[0][1].squeeze(0), encoder_hidden[1][1].squeeze(0))
        )

    #teacher forcingを使用する割合
    teacher_forcing_ratio = 0.5

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # teacher forcing使用
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_inputs[di], decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, decoder_inputs[di+1])
    else:
        '''
        decoder_inputsはすでにteacher_forcingを使用した状態であり，
        teacher_forcingを使わない場合にdecoder_inputsを書き換えている
        '''
        decoder_input = decoder_inputs[0]
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, decoder_inputs[di+1])

            _, topi = decoder_output.topk(1)  # (b,outdim) -> (b,1)
            decoder_input = topi.squeeze(1).detach()

    loss.backward()
    #↑lossはdouble型ではなくVariableクラスになっている
    #backwardメソッドを呼ぶことで逆伝搬がスタート，直前のノードに微分値をセット

    #エンコーダおよびデコーダの学習（パラメータの更新）
    encoder_optimizer.step()
    decoder_optimizer.step()

    #出力が可変長なのでlossも1ノードあたりに正規化
    return loss.item() / target_length



#1バッチデータあたりのバリデーション
def batch_valid(X, Y, encoder, decoder, criterion, lang):
    with torch.no_grad():
        '''
        X : (s, b)
        Y : (s, b)
        '''
        batch_size = X.size(1)
        target_length = Y.size(0)
        Y = Y[:target_length]

        loss = 0

        encoder_outputs, encoder_hidden = encoder(X)  # (s, b, 2h), ((1, b, h), (1, b, h))
        decoder_input = torch.tensor([SOS_token] * batch_size, device=my_device)  # (b)
        decoder_hidden = (
            (encoder_hidden[0][0].squeeze(0), encoder_hidden[1][0].squeeze(0)),
            (encoder_hidden[0][1].squeeze(0), encoder_hidden[1][1].squeeze(0))
            )

        decoded_outputs = torch.zeros(target_length, batch_size, lang.n_words, device=my_device)
        decoded_words = torch.zeros(batch_size, target_length, device=my_device)

        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)  # (b,odim), ((b,h),(b,h)), (b,il)
            decoded_outputs[di] = decoder_output

            loss += criterion(decoder_output, Y[di])

            _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)
            decoded_words[:, di] = topi[:, 0]  # (b)
            decoder_input = topi.squeeze(1)

        #出力が可変長なのでlossも1ノードあたりに正規化
        return loss.item() / target_length


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
def trainIters(lang, encoder, decoder, train_pairs, val_pairs, n_iters, print_every=10, learning_rate=0.01, saveModel=False):
    print("Training...")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0

    plot_val_losses = []
    print_val_loss_total = 0  # Reset every print_every
    plot_val_loss_total = 0

    best_val_loss=1000000   #仮
    best_iter=0

    best_encoder_weight = copy.deepcopy(encoder.state_dict())
    best_decoder_weight = copy.deepcopy(decoder.state_dict())

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    X_train=[pad_indexes(lang, s, dev=my_CPU) for s in train_pairs[0]]
    y_train=[pad_indexes(lang, s, dev=my_CPU) for s in train_pairs[1]]
    X_val=[pad_indexes(lang, s, dev=my_CPU) for s in val_pairs[0]]
    y_val=[pad_indexes(lang, s, dev=my_CPU) for s in val_pairs[1]]

    train_data_num=len(X_train)
    val_data_num=len(X_val)

    X_train=torch.tensor(X_train, dtype=torch.long, device=my_CPU)
    y_train=torch.tensor(y_train, dtype=torch.long, device=my_CPU)
    X_val=torch.tensor(X_val, dtype=torch.long, device=my_CPU)
    y_val=torch.tensor(y_val, dtype=torch.long, device=my_CPU)

    ds_train = TensorDataset(X_train, y_train)
    ds_val = TensorDataset(X_val, y_val)

    loader_train = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True)
    loader_val = DataLoader(ds_val, batch_size=BATCH_SIZE, shuffle=False)



    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for iter in range(1, n_iters + 1):
        for x, y in loader_train:
            '''
            x:(バッチサイズ, 文長)
            y:(バッチサイズ, 文長)
            からembedding層の入力に合うようにtransposeで入れ替え
            '''
            x=x.transpose(0,1)
            y=y.transpose(0,1)
            x=x.to(my_device)
            y=y.to(my_device)
            loss = batch_train(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            loss=loss*x.size(1)

            print_loss_total += loss
            plot_loss_total += loss
        #ここで学習1回分終わり

        for x, y in loader_val:
            x=x.transpose(0,1)
            y=y.transpose(0,1)
            x=x.to(my_device)
            y=y.to(my_device)
            val_loss = batch_valid(x, y, encoder, decoder, criterion, lang)

            val_loss=val_loss*x.size(1)

            print_val_loss_total += val_loss
            plot_val_loss_total += val_loss

        #画面にlossと時間表示
        #経過時間 (- 残り時間) (現在のiter 進行度) loss val_loss
        if iter == 1:
            print('%s (%d %d%%) loss=%.4f, val_loss=%.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_total, print_val_loss_total))

        elif iter % print_every == 0:
            print_loss_avg = (print_loss_total/train_data_num) / print_every
            print_loss_total = 0
            print_val_loss_avg = (print_val_loss_total/val_data_num) / print_every
            print_val_loss_total = 0
            print('%s (%d %d%%) loss=%.4f, val_loss=%.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, print_val_loss_avg))

        #lossグラフ記録
        plot_loss_avg = plot_loss_total/train_data_num
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        plot_val_loss_avg = plot_val_loss_total/val_data_num
        plot_val_losses.append(plot_val_loss_avg)
        plot_val_loss_total = 0

        #val_loss最小更新
        if (best_val_loss > val_loss) or (iter == 1):
            best_val_loss = val_loss
            best_iter=iter
            best_encoder_weight = copy.deepcopy(encoder.state_dict())
            best_decoder_weight = copy.deepcopy(decoder.state_dict())

    #全学習終わり
    #lossグラフ描画
    showPlot2(plot_losses, plot_val_losses)

    #val_loss最小のモデルロード
    encoder.load_state_dict(best_encoder_weight)
    decoder.load_state_dict(best_decoder_weight)
    print('best iter='+str(best_iter))

    if saveModel:
        torch.save(encoder.state_dict(), save_path+'encoder_'+str(best_iter)+'.pth')
        torch.save(decoder.state_dict(), save_path+'decoder_'+str(best_iter)+'.pth')

    return encoder, decoder


#グラフの描画（画像ファイル保存）
def showPlot(loss, val_loss):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(loss, color='blue', marker='o', label='loss')
    plt.plot(val_loss, color='green', marker='o', label='val_loss')
    plt.savefig(save_path+'loss.png')

def showPlot2(loss, val_loss):
    plt.plot(loss, color='blue', marker='o', label='loss')
    plt.plot(val_loss, color='green', marker='o', label='val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path+'loss.png')


###########################
# 5.モデルによる予測
###########################

# 1データに対する予測
def evaluate(lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
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



'''
空所の予測方法2つある？
1（共通）.空所が始まるまで（「{」のところまでは正答文そのまま、「{」から予測スタート）
2a.「}」が出るまでデコーダに予測させる
2b.答えを見て空所内の単語数は固定

ひとまず2aの方でevaluate_cloze()は実装してる
'''
#空所内のみを予想
#evaluate()の拡張
def evaluate_cloze(lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
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

        tmp_list=normalizeString(sentence).split(' ')
        tmp_list.append('<EOS>')
        cloze_start=tmp_list.index('{')
        cloze_end=tmp_list.index('}')
        cloze_flag=0
        cloze_ct=0

        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)
            decoder_attentions.append(attention)

            #空所が始まるまでは空所外の部分はそのまま用いる
            #ここではEOSを考慮しなくてよい
            if di <= cloze_start:
                decoded_words.append(tmp_list[di])
                decoder_input = torch.tensor([input_indexes[di]], device=my_device)

            #空所内の予測
            elif cloze_flag == 0:
                _, topi = decoder_output.topk(1)  # (1, 1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    #EOSならば終了
                    break
                else:
                    word=lang.index2word[topi.item()]
                    decoded_words.append(word)
                    decoder_input = topi[0]
                    if word == '}':
                        cloze_flag=1
                    else:
                        cloze_ct+=1

            #空所後の予測
            else:
                word=tmp_list[di-cloze_ct]
                decoded_words.append(word)
                if word == '<EOS>':
                    break
                else:
                    decoder_input = torch.tensor([input_indexes[di-cloze_ct]], device=my_device)

        decoder_attentions = torch.cat(decoder_attentions, dim=0)  # (l, n)

        #返り値は予測した単語列とattentionの重み？
        return decoded_words, decoder_attentions.squeeze(0)

#前方一致の確認
def forward_match(words, cloze_words, cloze_ct):
    flag=1
    if len(words) >= cloze_ct:
        for i in range(cloze_ct):
            if not  words[i] == cloze_words[i]:
                flag=0
        if flag == 1:
            return True

    return False


#これまでの予測と選択肢から次の１語候補リストを作成
def make_next_word(cloze_ct, cloze_words, choices):
    next_word_list=[]

    for words in choices:
        words_list=words.split(' ')
        if cloze_ct==0:
            next_word_list.append(words_list[0])
        else:
            #x番目を予測するときｘ−１番目まで一致しているなら
            if forward_match(words_list, cloze_words, cloze_ct):
                if len(words_list) == cloze_ct:
                    #その選択肢が終わりの時
                    next_word_list.append('}')
                elif len(words_list) > cloze_ct:
                    #その選択肢の次の1語を格納
                    next_word_list.append(words_list[cloze_ct])
    if next_word_list:
        #pythonではlistが空でなければTrue
        #重複を削除
        next_word_list=list(set(next_word_list))
    else:
        #TODO そもそもこのケースある？
        next_word_list.append('}')

    return next_word_list


#候補リストから確率最大の1語を返す
def pred_next_word(lang, next_word_list, decoder_output_data):
    if len(next_word_list)==1:
        max_word=next_word_list[0]
    else:
        max_p=decoder_output_data.min().item()
        for word in next_word_list:
            index=lang.check_word2index(word)
            p=decoder_output_data[0][index].item()
            if max_p < p:
                max_p = p
                max_word=word

    return max_word


#空所内のみを予想かつ選択肢の利用
#evaluate_clozeの拡張
def evaluate_choice(lang, encoder, decoder, sentence, choices, max_length=MAX_LENGTH):
    with torch.no_grad():
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

        tmp_list=normalizeString(sentence).split(' ')
        tmp_list.append('<EOS>')
        cloze_start=tmp_list.index('{')
        cloze_end=tmp_list.index('}')
        cloze_flag=0
        cloze_ct=0
        cloze_words=[]

        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)
            decoder_attentions.append(attention)

            #空所が始まるまでは空所外の部分はそのまま用いる
            #ここではEOSを考慮しなくてよい
            if di <= cloze_start:
                decoded_words.append(tmp_list[di])
                decoder_input = torch.tensor([input_indexes[di]], device=my_device)

            #空所内の予測
            # } までdecorded_wordに格納
            elif cloze_flag == 0:
                #これまでの予測と選択肢から次の１語候補リストを作成
                next_word_list=make_next_word(cloze_ct, cloze_words, choices)
                #候補リストから確率最大の1語を返す
                word=pred_next_word(lang, next_word_list, decoder_output.data)
                cloze_words.append(word)
                decoded_words.append(word)
                word_tensor=torch.tensor([lang.check_word2index(word)], device=my_device)
                decoder_input = word_tensor

                if word == '}':
                    cloze_flag=1
                else:
                    cloze_ct+=1

            #空所後の予測
            else:
                word=tmp_list[di-cloze_ct]
                decoded_words.append(word)
                if word == '<EOS>':
                    break
                else:
                    decoder_input = torch.tensor([input_indexes[di-cloze_ct]], device=my_device)

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


#文章からn-gramの集合を作成
def get_ngrams(segment, max_order):
    ngram_counts = collections.Counter()
    for order in range(1, max_order + 1):
        for i in range(0, len(segment) - order + 1):
          ngram = tuple(segment[i:i+order])
          ngram_counts[ngram] += 1
    return ngram_counts


def compute_bleu(preds_sentences, ans_sentences, max_order=4,
                 smooth=False):
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    pred_length = 0
    ans_length = 0
    for (preds, ans) in zip(preds_sentences, ans_sentences):
        pred_length += len(preds)
        ans_length += len(ans)

        merged_pred_ngram_counts = get_ngrams(preds, max_order)
        ans_ngram_counts = get_ngrams(ans, max_order)

        #2つのngram集合の積集合
        overlap = ans_ngram_counts & merged_pred_ngram_counts
        for ngram in overlap:
            matches_by_order[len(ngram)-1] += overlap[ngram]
        for order in range(1, max_order+1):
            possible_matches = len(ans) - order + 1
            if possible_matches > 0:
                possible_matches_by_order[order-1] += possible_matches

    precisions = [0] * max_order
    for i in range(0, max_order):
        if smooth:
            precisions[i] = ((matches_by_order[i] + 1.) /
                           (possible_matches_by_order[i] + 1.))
        else:
            if possible_matches_by_order[i] > 0:
                precisions[i] = (float(matches_by_order[i]) /
                             possible_matches_by_order[i])
            else:
                precisions[i] = 0.0

    if min(precisions) > 0:
        p_log_sum = sum((1. / max_order) * math.log(p) for p in precisions)
        geo_mean = math.exp(p_log_sum)
    else:
        geo_mean = 0

    if pred_length!=0:
        ratio = float(ans_length) / pred_length
        if ratio > 1.0:
            bp = 1.
        else:
            bp = math.exp(1 - 1. / ratio)
        bleu = geo_mean * bp
    else:
        ratio=0
        bp=0
        bleu=0

    return bleu


def is_correct_cloze(line):
    left=line.count('{')
    right=line.count('}')
    if left*right==1:
        return True

    return False


def get_cloze(line):
    line=re.sub(r'.*{ ', '', line)
    line=re.sub(r' }.*', '', line)

    return line


#部分一致判定用
def match(pred_cloze, ans_cloze):
    pred_set=set(pred_cloze.split(' '))
    ans_set=set(ans_cloze.split(' '))
    i=0

    for word in pred_set:
        if word in ans_set:
            i+=1

    return i


#精度いろいろ計算
#問題文、完全一致文、空所の完答文、空所の一部正答文、BLEU値、空所ミス文
def calc_score(preds_sentences, ans_sentences):
    line_num=0
    allOK=0
    clozeOK=0
    partOK=0
    miss=0

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

    BLEU=compute_bleu(preds_sentences, ans_sentences)

    return line_num, allOK, clozeOK, partOK, BLEU, miss


def output_preds(file_name, preds):
    with open(file_name, 'w') as f:
        for p in preds:
            f.write(p+'\n')


def print_score(line, allOK, clozeOK, partOK, BLEU, miss):
    print('  acc(all): ', '{0:.2f}'.format(1.0*allOK/line*100),' %')
    print('acc(cloze): ', '{0:.2f}'.format(1.0*clozeOK/line*100),' %')
    print(' acc(part): ', '{0:.2f}'.format(1.0*partOK/line*100),' %')

    print(' BLEU: ','{0:.2f}'.format(BLEU*100.0))
    print('  all: ', allOK)
    print('cloze: ',clozeOK)
    print(' part: ',partOK)
    print(' line: ',line)
    print(' miss: ',miss)


def output_score(file_name, line, allOK, clozeOK, partOK, BLEU, miss):
    output=''
    output=output+'  acc(all): '+str(1.0*allOK/line*100)+' %\n'
    output=output+'acc(cloze): '+str(1.0*clozeOK/line*100)+' %\n'
    output=output+' acc(part): '+str(1.0*partOK/line*100)+' %\n\n'
    output=output+'      BLEU: '+str(BLEU*100.0)+' %\n\n'
    output=output+'       all: '+str(allOK)+'\n'
    output=output+'     cloze: '+str(clozeOK)+'\n'
    output=output+'      part: '+str(partOK)+'\n'
    output=output+'      line: '+str(line)+'\n'
    output=output+'      miss: '+str(miss)+'\n'

    with open(file_name, 'w') as f:
        f.write(output)


def get_choices(file_name):
    print("Reading data...")
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=get_cloze(normalizeString(line, choices=True))
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


def score(preds, ans, file_output, file_name):
    #精度のprintとファイル出力
    line, allOK, clozeOK, partOK, BLEU, miss = calc_score(preds, ans)
    print_score(line, allOK, clozeOK, partOK, BLEU, miss)
    if file_output:
        output_score(file_name, line, allOK, clozeOK, partOK, BLEU, miss)


#テストデータに対する予測と精度計算
#空所内のみを予測するモード
#および、選択肢を利用するモード
def test_choices(lang, encoder, decoder, test_data, choices, saveAttention=False, file_output=False):
    print("Test ...")
    #input_sentence や ansは文字列であるのに対し、output_wordsはリストであることに注意
    preds=[]
    ans=[]
    preds_cloze=[]
    preds_choices=[]
    for pair, choi in zip(test_data, choices):
        input_sentence=pair[0]
        ans.append(pair[1])

        output_words, attentions = evaluate(lang, encoder, decoder, input_sentence)
        preds.append(' '.join(output_words))

        output_cloze_ct, cloze_attentions = evaluate_cloze(lang, encoder, decoder, input_sentence)
        preds_cloze.append(' '.join(output_cloze_ct))

        output_choice_words, choice_attentions = evaluate_choice(lang, encoder, decoder, input_sentence, choi)
        preds_choices.append(' '.join(output_choice_words))

        if saveAttention:
            showAttention('all', input_sentence, output_words, attentions)
            showAttention('cloze', input_sentence, output_cloze_ct, cloze_attentions)
            showAttention('choice', input_sentence, output_choice_words, choice_attentions)
        if file_output:
            output_preds(save_path+'preds.txt', preds)
            output_preds(save_path+'preds_cloze.txt', preds_cloze)
            output_preds(save_path+'preds_choices.txt', preds_choices)
    print("Calc scores ...")
    score(preds, ans, file_output, save_path+'score.txt')
    score(preds_cloze, ans, file_output, save_path+'score_cloze.txt')
    score(preds_choices, ans, file_output, save_path+'score_choices.txt')


#選択肢を使って4つの文を生成
def make_sents_with_cloze_mark(sentence, choices):
    sents=[]
    before=re.sub(r'{.*', '{ ', sentence)
    after=re.sub(r'.*}', ' }', sentence)
    for choice in choices:
        tmp=before + choice + after
        sents.append(tmp.strip())

    return sents

#1文に対して文スコアを算出
def calc_sent_score(lang, encoder, decoder, sent, max_length=MAX_LENGTH):
    #evaluate_choiceから改変
    score=0
    with torch.no_grad():
        input_indexes = pad_indexes(lang, sent)
        input_batch = torch.tensor([input_indexes], dtype=torch.long, device=my_device)  # (1, s)

        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1))
        decoder_input = torch.tensor([SOS_token], device=my_device)  # SOS
        decoder_hidden = (
            (encoder_hidden[0][0].squeeze(0), encoder_hidden[1][0].squeeze(0)),
            (encoder_hidden[0][1].squeeze(0), encoder_hidden[1][1].squeeze(0))
            )


        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)

            score+=decoder_output.data[0][input_indexes[di]]

            if input_indexes[di] == EOS_token:
                break
            decoder_input = torch.tensor([input_indexes[di]], device=my_device)

    return score/len(sent.split(' '))



def get_best_sent(lang, encoder, decoder, sents):
    scores=[]
    for sent in sents:
        score=calc_sent_score(lang, encoder, decoder, sent)
        scores.append(score)

    #scoreが最大の分を返す
    #numpyへの変換考えるとこっちのほうが速い？
    return sents[scores.index(max(scores))]

#一旦1語以上，選択肢ありモード
#TODO あとで全単語からもできるように
def test_choices_by_sent_score(lang, encoder, decoder, test_data, choices, saveAttention=False, file_output=False):
    print("Test by sent score...")
    #input_sentence や ansは文字列であるのに対し、output_wordsはリストであることに注意
    preds=[]
    ans=[]
    preds_cloze=[]
    preds_choices=[]
    for pair, choi in zip(test_data, choices):
        input_sentence=pair[0]
        ans.append(pair[1])

        sents=make_sents_with_cloze_mark(input_sentence, choi)
        pred=get_best_sent(lang, encoder, decoder, sents)

        preds_choices.append(pred)

        if file_output:
            output_preds(save_path+'preds_choices.txt', preds_choices)
    print("Calc scores ...")
    score(preds_choices, ans, file_output, save_path+'score_choices.txt')


#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--encoder', help='encoder file name (when load model, mode=test)')
    parser.add_argument('--decoder', help='decoder file name (when load model, mode=test)')
    parser.add_argument('--epoch', type=int, default=30)
    #TODO ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()
    print(args.mode)

    # 1.語彙データ読み込み
    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)

    # 2.モデル定義
    weights_matrix=get_weight_matrix(vocab)
    my_encoder = EncoderRNN(vocab.n_words, EMB_DIM, HIDDEN_DIM, weights_matrix).to(my_device)
    my_decoder = AttnDecoderRNN2(EMB_DIM, HIDDEN_DIM, ATTN_DIM, vocab.n_words, weights_matrix).to(my_device)

    #学習時
    if args.mode == 'all' or args.mode == 'mini':
        #train_cloze=file_path+'tmp_cloze.txt'
        #train_ans=file_path+'tmp_ans.txt'

        #text8全体
        train_cloze=file_path+'text8_cloze.txt'
        train_ans=file_path+'text8_ans.txt'
        '''
        #enwiki1GB
        train_cloze='/media/tamaki/HDCL-UT/tamaki/M2/data_for_seq2seq/enwiki1GB_seq2seq_c.txt'
        train_ans='/media/tamaki/HDCL-UT/tamaki/M2/data_for_seq2seq/enwiki1GB_seq2seq_ans.txt'
        '''
        if args.mode == 'mini':
            #合同ゼミ
            train_cloze=file_path+'text8_cloze50000.txt'
            train_ans=file_path+'text8_ans50000.txt'

        #all_data=readData(train_cloze, train_ans)
        print("Reading data...")
        all_X=readData2(train_cloze)
        all_Y=readData2(train_ans)


        if args.mode == 'mini':
            #all_data=all_data[:20]
            all_X=all_X[:20]
            all_Y=all_Y[:20]

        #train_data, val_data = train_test_split(all_data, test_size=0.1)
        train_X, val_X = train_test_split(all_X, test_size=0.1)
        train_Y, val_Y = train_test_split(all_Y, test_size=0.1)

        train_data = (train_X, train_Y)
        val_data = (val_X, val_Y)

        #モデルとか結果とかを格納するディレクトリの作成
        save_path=save_path+args.mode+'_seq2seq'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'

        # 3.学習
        my_encoder, my_decoder = trainIters(vocab, my_encoder, my_decoder, train_data, val_data, n_iters=args.epoch, saveModel=True)

    #すでにあるモデルでテスト時
    else:
        save_path=args.model_dir+'/'

        my_encoder.load_state_dict(torch.load(save_path+args.encoder))
        my_decoder.load_state_dict(torch.load(save_path+args.decoder))

        save_path=save_path+today_str

    # 4.評価
    center_cloze=git_data_path+'center_cloze.txt'
    center_ans=git_data_path+'center_ans.txt'
    center_choi=git_data_path+'center_choices.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'
    MS_choi=git_data_path+'microsoft_choices.txt'

    print("Reading data...")
    center_data=readData(center_cloze, center_ans)
    center_choices=get_choices(center_choi)

    MS_data=readData(MS_cloze, MS_ans)
    MS_choices=get_choices(MS_choi)

    if args.mode == 'mini' or args.mode == 'mini_test':
        center_data=center_data[:5]
        center_choices=center_choices[:5]
        MS_data=MS_data[:5]
        MS_choices=MS_choices[:5]


    #テストデータに対する予測と精度の計算
    #選択肢を使ったテスト
    #これは前からの予測
    print('center')
    test_choices(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=True)

    #これは文スコア
    test_choices_by_sent_score(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=False)

    print('MS')
    test_choices(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=True)

    #これは文スコア
    test_choices_by_sent_score(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=False)
