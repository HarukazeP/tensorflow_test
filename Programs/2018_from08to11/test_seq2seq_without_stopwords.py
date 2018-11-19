# -*- coding: utf-8 -*-

'''
check_model_output_top10_words.py から変更
学習はせずテストのみ
テスト時にストップワードやUNKを除いて予測

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

from nltk.corpus import stopwords

#----- グローバル変数一覧 -----
MAX_LENGTH = 40
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
SOS_token = 1
EOS_token = 2
UNK_token = 3

#事前処理いろいろ
print('Start: '+today_str)
if torch.cuda.is_available():
    my_device = torch.device("cuda")
    print('Use GPU')
else:
    my_device= torch.device("cpu")

stopword_set = set(stopwords.words('english'))
ex_word=set(['one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'zero', '{', '}', 'SOS', 'EOS', '<UNK>'])
stopword_set |= ex_word #和集合

print('stopwords set', stopword_set)

'''
# TODO: なんかlinkgrammerのインストールできない
en_dir = linkgrammar.Dictionary()
po = linkgrammar.ParseOptions(min_null_count=0, max_null_count=999)
'''
#----- 関数群 -----


###########################
# 1.データの準備
###########################

#seq2seqモデルで用いる語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": UNK_token}
        self.word2count = {"<UNK>": 0}
        self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS", UNK_token: "<UNK>"}
        self.n_words = 4  # PAD と SOS と EOS と UNK

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

#エンコーダのクラス
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, weights_matrix):
        super(EncoderRNN, self).__init__()
        self.input_dim = input_dim #入力語彙数
        self.embedding_dim = emb_dim
        self.hidden_dim = hid_dim

        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim, padding_idx=PAD_token) #語彙数×次元数
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))

        self.lstm = nn.LSTM(input_size=self.embedding_dim,
                            hidden_size=self.hidden_dim,
                            bidirectional=True,
                            num_layers=2)
        self.linear_h1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_c1 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

        self.linear_h2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        self.linear_c2 = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, input_batch):
        """
        :param input_batch: (s, b)

        :returns (s, b, 2h), ((1, b, h), (1, b, h))
        decoderの入力として使うのでLTSMの出力からいろいろ変形してる
        """

        batch_size = input_batch.shape[1]

        embedded = self.embedding(input_batch)  # (s, b) -> (s, b, h)
        output, (all_h, all_c) = self.lstm(embedded)

        hidden_h1, hidden_h2=torch.chunk(all_h, 2, dim=0)
        hidden_c1, hidden_c2=torch.chunk(all_c, 2, dim=0)


        hidden_h1 = hidden_h1.transpose(1, 0)  # (2, b, h) -> (b, 2, h)
        hidden_h1 = hidden_h1.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_h1 = F.dropout(hidden_h1, p=0.5, training=self.training)
        hidden_h1 = self.linear_h1(hidden_h1)  # (b, 2h) -> (b, h)
        hidden_h1 = F.relu(hidden_h1)
        hidden_h1 = hidden_h1.unsqueeze(0)  # (b, h) -> (1, b, h)

        hidden_c1 = hidden_c1.transpose(1, 0)
        hidden_c1 = hidden_c1.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_c1 = F.dropout(hidden_c1, p=0.5, training=self.training)
        hidden_c1 = self.linear_c1(hidden_c1)
        hidden_c1 = F.relu(hidden_c1)
        hidden_c1 = hidden_c1.unsqueeze(0)  # (b, h) -> (1, b, h)


        hidden_h2 = hidden_h2.transpose(1, 0)  # (2, b, h) -> (b, 2, h)
        hidden_h2 = hidden_h2.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_h2 = F.dropout(hidden_h2, p=0.5, training=self.training)
        hidden_h2 = self.linear_h2(hidden_h2)  # (b, 2h) -> (b, h)
        hidden_h2 = F.relu(hidden_h2)
        hidden_h2 = hidden_h2.unsqueeze(0)  # (b, h) -> (1, b, h)

        hidden_c2 = hidden_c2.transpose(1, 0)
        hidden_c2 = hidden_c2.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_c2 = F.dropout(hidden_c2, p=0.5, training=self.training)
        hidden_c2 = self.linear_c2(hidden_c2)
        hidden_c2 = F.relu(hidden_c2)
        hidden_c2 = hidden_c2.unsqueeze(0)  # (b, h) -> (1, b, h)

        hidden_h = (hidden_h1, hidden_h2)
        hidden_c = (hidden_c1, hidden_c2)


        return output, (hidden_h, hidden_c)
        # (s, b, 2h), (((1, b, h), (1, b, h)), ((1, b, h), (1, b, h)))


'''
nn.LSTMもnn.LSTMCellも
入力：input,  h, c
出力：output, h, c
ではあるが次元が違う

nn.LSTMでは
input=(seq_len, batch_size, input_dim)
h=(num_layers*direction, batch_size, output_dim)

nn.LSTMCellでは
input=(batch_size, input_dim)
h=(batch_size, output_dim)


'''
#attentionつきデコーダのクラス
#attentionの形式をluongのやつに
class AttnDecoderRNN2(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_size, output_size, weights_matrix):
        super(AttnDecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=PAD_token)
        self.embedding.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.lstm = nn.LSTMCell(emb_size, hidden_size)
        self.lstm2 = nn.LSTMCell(hidden_size, hidden_size)

        self.score_w = nn.Linear(2*hidden_size, 2*hidden_size)
        self.attn_w = nn.Linear(4*hidden_size, attn_size)
        self.out_w = nn.Linear(attn_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param: input: (b)
        :param: hidden: ((b,h),(b,h))
        :param: encoder_outputs: (il,b,2h)

        :return: (b,o), ((b,h),(b,h)), (b,il)
        """

        embedded = self.embedding(input)  # (b) -> (b,e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)

        hidden1=hidden[0]
        hidden2=hidden[1]

        hidden1 = self.lstm(embedded, hidden1)  # (b,e),((b,h),(b,h)) -> ((b,h),(b,h))
        hidden2 = self.lstm2(hidden1[0], hidden2)  # (b,h),((b,h),(b,h)) -> ((b,h),(b,h))

        decoder_output = torch.cat(hidden2, dim=1)  # ((b,h),(b,h)) -> (b,2h)
        decoder_output = F.dropout(decoder_output, p=0.5, training=self.training)

        # score
        score = self.score_w(decoder_output)  # (b,2h) -> (b,2h)
        scores = torch.bmm(
            encoder_outputs.transpose(0, 1),  # (b,il,2h)
            score.unsqueeze(2)  # (b,2h,1)
        )  # (b,il,1)
        attn_weights = F.softmax(scores, dim=1)  # (b,il,1)

        # context
        context = torch.bmm(
            attn_weights.transpose(1, 2),  # (b,1,il)
            encoder_outputs.transpose(0, 1)  # (b,il,2h)
        )  # (b,1,2h)
        context = context.squeeze(1)  # (b,1,2h) -> (b,2h)

        concat = torch.cat((context, decoder_output), dim=1)  # ((b,2h),(b,2h)) -> (b,4h)
        #concat = F.dropout(concat, p=0.5, training=self.training)

        attentional = self.attn_w(concat)  # (b,4h) -> (b,a)
        attentional = torch.tanh(attentional)
        #attentional = F.dropout(attentional, p=0.5, training=self.training)

        output = self.out_w(attentional)  # (b,a) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, (hidden1, hidden2), attn_weights.squeeze(2)
        # (b,o), (((b,h),(b,h)), ((b,h),(b,h)), (b,il)


###########################
# 3.入力データ変換
###########################

#単語列をID列に
def indexesFromSentence(lang, sentence):
    return [lang.check_word2index(word) for word in sentence.split(' ')]


#単語列からモデルの入力へのテンソルに
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=my_device).view(-1, 1)


#単語列からモデルの入力へのテンソルに
#パディングあり、returnも変更
def pad_indexes(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    indexes + [0] * (MAX_LENGTH - len(indexes))
    return indexes + [0] * (MAX_LENGTH - len(indexes))


#入力と出力のペアからリストに
def indexesFromPair(lang, pair):
    input_tensor = pad_indexes(lang, pair[0])
    target_tensor = pad_indexes(lang, pair[1])
    return (input_tensor, target_tensor)




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

    decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                    (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )

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
        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )

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

    X_train=[pad_indexes(lang, s) for s in train_pairs[0]]
    y_train=[pad_indexes(lang, s) for s in train_pairs[1]]
    X_val=[pad_indexes(lang, s) for s in val_pairs[0]]
    y_val=[pad_indexes(lang, s) for s in val_pairs[1]]

    train_data_num=len(X_train)
    val_data_num=len(X_val)

    X_train=torch.tensor(X_train, dtype=torch.long, device=my_device)
    y_train=torch.tensor(y_train, dtype=torch.long, device=my_device)
    X_val=torch.tensor(X_val, dtype=torch.long, device=my_device)
    y_val=torch.tensor(y_val, dtype=torch.long, device=my_device)

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
            loss = batch_train(x, y, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

            loss=loss*x.size(1)

            print_loss_total += loss
            plot_loss_total += loss
        #ここで学習1回分終わり

        for x, y in loader_val:
            x=x.transpose(0,1)
            y=y.transpose(0,1)
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
        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )

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

        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )


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
def pred_next_word_without_stopwords(lang, next_word_list, decoder_output_data):
    max_word=''
    if len(next_word_list)==1:
        max_word=next_word_list[0]
    else:
        max_p=decoder_output_data.min().item()
        for word in next_word_list:
            index=lang.check_word2index(word)
            p=score_without_stopwords(lang, decoder_output_data[0], index)
            if max_p < p:
                max_p = p
                max_word=word

    return max_word



def count_up_top10_words(lang, top10_id, words_dict):
    for i in range(10):
        x=lang.index2word[top10_id[i].item()]
        if x in words_dict:
            words_dict[x]+=1
        else:
            words_dict[x]=1




#空所内のみを予想かつ選択肢の利用
#evaluate_clozeの拡張
def evaluate_choice_and_top10_words(lang, encoder, decoder, sentence, choices, words_dict,  max_length=MAX_LENGTH):
    with torch.no_grad():
        input_indexes = pad_indexes(lang, sentence)
        input_batch = torch.tensor([input_indexes], dtype=torch.long, device=my_device)  # (1, s)


        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1))

        decoder_input = torch.tensor([SOS_token], device=my_device)  # SOS

        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )


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
                word=pred_next_word_without_stopwords(lang, next_word_list, decoder_output.data)
                cloze_words.append(word)
                decoded_words.append(word)
                word_tensor=torch.tensor([lang.check_word2index(word)], device=my_device)
                decoder_input = word_tensor

                if word == '}':
                    cloze_flag=1
                else:
                    cloze_ct+=1
                    tmp=decoder_output[0]
                    min_p=tmp.min().item()-1
                    for word in stopword_set:
                        id=lang.check_word2index(word)
                        tmp[id]=min_p
                    _, top10 = tmp.topk(10)



                    count_up_top10_words(lang, top10, words_dict)

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
def test_choices_and_top10_words_without_stopwords(lang, encoder, decoder, test_data, choices, saveAttention=False, file_output=False):
    print("Test ...")
    #input_sentence や ansは文字列であるのに対し、output_wordsはリストであることに注意
    preds=[]
    ans=[]
    preds_cloze=[]
    preds_choices=[]
    words_d=dict()
    for pair, choi in zip(test_data, choices):
        input_sentence=pair[0]
        ans.append(pair[1])

        #output_words, attentions = evaluate(lang, encoder, decoder, input_sentence)
        #preds.append(' '.join(output_words))

        #output_cloze_ct, cloze_attentions = evaluate_cloze(lang, encoder, decoder, input_sentence)
        #preds_cloze.append(' '.join(output_cloze_ct))

        output_choice_words, choice_attentions = evaluate_choice_and_top10_words(lang, encoder, decoder, input_sentence, choi, words_d)
        preds_choices.append(' '.join(output_choice_words))

        if saveAttention:
            #showAttention('all', input_sentence, output_words, attentions)
            #showAttention('cloze', input_sentence, output_cloze_ct, cloze_attentions)
            showAttention('choice', input_sentence, output_choice_words, choice_attentions)
        if file_output:
            #output_preds(save_path+'preds.txt', preds)
            #output_preds(save_path+'preds_cloze.txt', preds_cloze)
            output_preds(save_path+'preds_choices.txt', preds_choices)
    print('model output top10 words')
    words_list = sorted(words_d.items(), key=lambda x: x[1], reverse=True)
    for i in range(10):
        print(words_list[i])
    print("Calc scores ...")
    #score(preds, ans, file_output, save_path+'score.txt')
    #score(preds_cloze, ans, file_output, save_path+'score_cloze.txt')
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

#from linkgrammar import Sentence, ParseOptions, Dictionary, Clinkgrammar as clg

def is_grammer_OK(text):
    sent=linkgrammar.Sentence(text, en_dir, po)
    linkages = sent.parse()

    if sent.null_count() >0:
        return False

    return True


def make_sents_with_cloze_mark_with_grammer(sentence, choices, raw_sentence, raw_choices):
    sents=[]
    flag=0
    raw_before=re.sub(r'{.*', '{ ', raw_sentence)
    raw_after=re.sub(r'.*}', ' }', raw_sentence)
    before=re.sub(r'{.*', '{ ', sentence)
    after=re.sub(r'.*}', ' }', sentence)
    for r_choi, choi in zip(raw_choices, choices):
        raw_tmp=raw_before + r_choi + raw_after
        tmp=before + choi + after
        if is_grammer_OK(raw_tmp.strip()):
            sents.append(tmp.strip())
            flag=1

    if flag==0:
        sents=make_sents_with_cloze_mark(sentence, choices)

    return sents

def score_without_stopwords(lang, decoder_output_data, word_id):
    min_p=decoder_output_data.min().item()

    for word in stopword_set:
        id=lang.check_word2index(word)
        decoder_output_data[id]=min_p

    return decoder_output_data[word_id]

#1文に対して文スコアを算出
def calc_sent_score(lang, encoder, decoder, sent, max_length=MAX_LENGTH):
    #evaluate_choiceから改変
    score=0
    with torch.no_grad():
        input_indexes = pad_indexes(lang, sent)
        input_batch = torch.tensor([input_indexes], dtype=torch.long, device=my_device)  # (1, s)

        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1))
        decoder_input = torch.tensor([SOS_token], device=my_device)  # SOS
        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )


        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)

            score+=decoder_output.data[0][input_indexes[di]]

            if input_indexes[di] == EOS_token:
                break
            decoder_input = torch.tensor([input_indexes[di]], device=my_device)

    return score/len(sent.split(' '))


#1文に対して文スコアを算出
def calc_sent_score_without_stopwords(lang, encoder, decoder, sent, max_length=MAX_LENGTH):
    #evaluate_choiceから改変
    score=0
    with torch.no_grad():
        input_indexes = pad_indexes(lang, sent)
        input_batch = torch.tensor([input_indexes], dtype=torch.long, device=my_device)  # (1, s)

        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1))
        decoder_input = torch.tensor([SOS_token], device=my_device)  # SOS
        decoder_hidden = ( (encoder_hidden[0][0].squeeze(0), encoder_hidden[0][1].squeeze(0)),
                (encoder_hidden[1][0].squeeze(0), encoder_hidden[1][1].squeeze(0)) )


        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(decoder_input, decoder_hidden, encoder_outputs)  # (1,outdim), ((1,h),(1,h)), (l,1)

            score+=score_without_stopwords(lang, decoder_output.data[0], input_indexes[di])

            if input_indexes[di] == EOS_token:
                break
            decoder_input = torch.tensor([input_indexes[di]], device=my_device)

    return score/len(sent.split(' '))



def get_best_sent(lang, encoder, decoder, sents, use_stopwords):
    scores=[]
    for sent in sents:
        if use_stopwords:
            score=calc_sent_score_without_stopwords(lang, encoder, decoder, sent)
        else:
            score=calc_sent_score(lang, encoder, decoder, sent)
        scores.append(score)

    #scoreが最大の分を返す
    #numpyへの変換考えるとこっちのほうが速い？
    return sents[scores.index(max(scores))]

#一旦1語以上，選択肢ありモード
#ストップワードの確率を最低にする
def test_choices_by_sent_score_without_stopwords(lang, encoder, decoder, test_data, choices, saveAttention=False, file_output=False):
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
        pred=get_best_sent(lang, encoder, decoder, sents, use_stopwords=True)

        preds_choices.append(pred)

        if file_output:
            output_preds(save_path+'preds_choices.txt', preds_choices)
    print("Calc scores ...")
    score(preds_choices, ans, file_output, save_path+'score_choices.txt')


#一旦1語以上，選択肢ありモード
#ストップワードの確率を最低にする
def test_choices_by_sent_score_with_grammar(lang, encoder, decoder, test_data, choices, raw_data, raw_choices, saveAttention=False, file_output=False, use_stopwords=False):
    print("Test by sent score...")
    #input_sentence や ansは文字列であるのに対し、output_wordsはリストであることに注意
    preds=[]
    ans=[]
    preds_cloze=[]
    preds_choices=[]
    for pair, choi, raw_cloze, raw_choi in zip(test_data, choices, raw_data, raw_choices):
        input_sentence=pair[0]
        ans.append(pair[1])

        sents=make_sents_with_cloze_mark_with_grammer(input_sentence, choi, raw_cloze, raw_choi)
        pred=get_best_sent(lang, encoder, decoder, sents, use_stopwords=use_stopwords)

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
    vocab0=git_data_path+'enwiki_vocab30000.txt'
    vocab1=git_data_path+'enwiki_vocab10000.txt'
    vocab2=git_data_path+'enwiki_vocab1000.txt'
    vocab3=git_data_path+'enwiki_vocab100.txt'


    encoder0='encoder_15.pth'
    decoder0='decoder_15.pth'
    dir0='11_04_2242all_seq2seq_vocab30000'

    encoder1='encoder_15.pth'
    decoder1='decoder_15.pth'
    dir1='11_15_1757all_seq2seq_vocab10000'

    encoder2='encoder_16.pth'
    decoder2='decoder_16.pth'
    dir2='11_14_1820all_seq2seq_vocab1000'

    encoder3='encoder_14.pth'
    decoder3='decoder_14.pth'
    dir3='11_15_0604all_seq2seq_vocab100'

    files=[(vocab0, encoder0, decoder0, dir0),
            (vocab1, encoder1, decoder1, dir1),
            (vocab2, encoder2, decoder2, dir2),
            (vocab3, encoder3, decoder3, dir3)]

    for model in files:
        today1=datetime.datetime.today()
        today_str=today1.strftime('%m_%d_%H%M')
        save_path=file_path+model[3]+'/'+today_str

        # 1.語彙データ読み込み
        vocab = readVocab(model[0])

        # 2.モデル定義
        #テスト時なのでembedding初期値読み込まなくていい
        weights_matrix=np.zeros((vocab.n_words, EMB_DIM))
        my_encoder = EncoderRNN(vocab.n_words, EMB_DIM, HIDDEN_DIM, weights_matrix).to(my_device)
        my_decoder = AttnDecoderRNN2(EMB_DIM, HIDDEN_DIM, ATTN_DIM, vocab.n_words, weights_matrix).to(my_device)

        my_encoder.load_state_dict(torch.load(file_path+model[3]+'/'+model[1]))
        my_decoder.load_state_dict(torch.load(file_path+model[3]+'/'+model[2]))

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


        #テストデータに対する予測と精度の計算
        #選択肢を使ったテスト
        #これは前からの予測
        print(model[0])
        print('center')
        print('without stopwords')
        test_choices_and_top10_words_without_stopwords(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=False)

        #これは文スコア
        #print('without stopwords')
        #test_choices_by_sent_score_without_stopwords(vocab, my_encoder, my_decoder, center_data, center_choices, saveAttention=False, file_output=False)
        #test_choices_by_sent_score_with_grammar(vocab, my_encoder, my_decoder, center_data, center_choices, center_raw_data, center_raw_choices, saveAttention=False, file_output=False, use_stopwords=False)
        #test_choices_by_sent_score_with_grammar(vocab, my_encoder, my_decoder, center_data, center_choices, center_raw_data, center_raw_choices, saveAttention=False, file_output=False, use_stopwords=True)

        print('MS')
        test_choices_and_top10_words_without_stopwords(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=False)

        #これは文スコア
        #print('without stopwords')
        #test_choices_by_sent_score_without_stopwords(vocab, my_encoder, my_decoder, MS_data, MS_choices, saveAttention=False, file_output=False)
        #test_choices_by_sent_score_with_grammar(vocab, my_encoder, my_decoder, MS_data, MS_choices, MS_raw_data, MS_raw_choices, saveAttention=False, file_output=False, use_stopwords=False)
        #test_choices_by_sent_score_with_grammar(vocab, my_encoder, my_decoder, MS_data, MS_choices, MS_raw_data, MS_raw_choices, saveAttention=False, file_output=False, use_stopwords=True)
