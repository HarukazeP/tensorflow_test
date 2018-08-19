# -*- coding: utf-8 -*-
'''
もともとは英語でコメント書かれていたやつを日本語にしたり
変数ｍ理買えたりとか

#TODO いわゆるmain部的な整理を合同ゼミ後


入力データは
./data/xxx-yyy.txt
xxxが翻訳前，yyy翻後の言語
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



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#自分で定義したグローバル関数とか
data_path='./data/wiki_tail20.txt'
file_path='./data/'
today1=datetime.datetime.today()
today_str=file_path + today1.strftime('%m_%d_%H%M')


###########################
# 1.データの準備
###########################

SOS_token = 0
EOS_token = 1

#seq2seqモデルで用いる語彙に関するクラス
class Lang:
    #nameは言語の種類というか区別用
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
    
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

#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

#データの前処理
#strip()は文頭文末の改行や空白を取り除いてくれる
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r'[^a-zA-Z{}]', ' ', s)
    s = re.sub(r'[ ]+', ' ', s)
    
    return s


#ファイルからデータ読み込み
#引数reverseは翻訳の向きを変更，英→仏から仏→英へみたいな
def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    #データを一括読み込みし行ごとに分割
    #データの末尾の改行は削除
    lines = open(data_path, encoding='utf-8').\
        read().strip().split('\n')

    #入力データと教師データのペアの作成
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    #語彙の作成
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs





#↓自分で変更したやつ
#一応センターの最長が36だったから
MAX_LENGTH = 40


#データの読み込みから語彙のカウントまで
def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    return input_lang, output_lang, pairs


#データの一例の確認
input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
print(random.choice(pairs))




###########################
# 2.モデル定義
###########################

#クラスのこととかよく分かってないのでコメント過剰気味



#エンコーダのクラス
#nn.Moduleクラスを継承
class EncoderRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        #__init__(～～)はクラスを呼び出した時に自動で呼び出されるやつ
        super(EncoderRNN, self).__init__()
        #superによって親クラスのinitを呼び出せる
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(input_dim, hidden_dim) #語彙数×次元数
        self.gru = nn.GRU(input_size=hidden_dim, hidden_size=hidden_dim)

    def forward(self, input, hidden_0):
        embedded = self.embedding(input).view(1, 1, -1)
        #viewはnumpyのreshape的な
        # -1は自動調整
        gru_in = embedded
        output, hidden = self.gru(gru_in, hidden_0)
        #GRUはRNNの一種なので出力が2つ，t-1的な
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)

'''
もしembeddingの初期値与えたかったら

embed = nn.Embedding(num_embeddings, embedding_dim)
# pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
embed.weight.data.copy_(torch.from_numpy(pretrained_weight))



'''




#attentionつきデコーダのクラス
class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_dim, output_dim, dropout_p=0.1, max_length=MAX_LENGTH):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p    #ドロップアウト率
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_dim, self.hidden_dim)
        self.attn = nn.Linear(self.hidden_dim * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_dim * 2, self.hidden_dim)
        #Linearは全結合層
        #引数は(入力ユニット数, 出力ユニット数)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_dim, self.hidden_dim)
        self.out = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, input, hidden_0, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_input= torch.cat((embedded[0], hidden_0[0]), dim=1)
        #torch.catはテンソルの連結
        attn_weights = F.softmax(self.attn(attn_input), dim=1)
        #softmaxだと合計が1になるような計算をするが，どの次元で見るかというのがdim

        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))
        #torch.bmmは2つのテンソルの積みたいな感じ
        #(b×n×m)と(b×m×p)の形でなければならない
        #unsqueeze(0)によって0次元目に次元を追加，テンソルの変形的な

        gru_in = torch.cat((embedded[0], attn_applied[0]), 1)
        gru_in = self.attn_combine(gru_in).unsqueeze(0)

        gru_in = F.relu(gru_in)
        output, hidden = self.gru(gru_in, hidden_0)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_dim, device=device)





#TODO　ここから下読み込んで理解する

#次は学習データの準備



###########################
# 3.入力データ変換
###########################


#単語列をID列に
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

#単語列からモデルの入力へのテンソルに
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

#入力と出力のペアからテンソルに
def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)




###########################
# 4.モデルの学習
###########################

'''
モデルの訓練

“Teacher forcing” は(seq2seqのでの)次の入力としてデコーダの推測を使用する代わりに、実際のターゲット出力を各次の入力として使用する概念です。

PyTorch autograd が与えてくれる自由度ゆえに、単純な if ステートメントで "teacher forcing" を使用するか否かをランダムに選択することができます。それを更に使用するためには teacher_forcing_ratio を上向きに調整してください。
'''



#teacher forcingを使用する割合
teacher_forcing_ratio = 0.5

'''
引数
input_tensor:      入力テンソル
target_tensor:     教師テンソル
encoder:           エンコーダのクラス
decoder:           デコーダのクラス
encoder_optimizer: エンコーダの最適化クラス
decoder_optimizer: デコーダの最適化クラス
criterion:         誤差の計算手法クラス
max_length:        入力および教師データの最大長(最大単語数)

'''

def train(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

    loss = 0
    
    #エンコーダの準備
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]


    #デコーダの準備
    decoder_input = torch.tensor([[SOS_token]], device=device)

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # teacher forcing使用
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_tensor[di])
            decoder_input = target_tensor[di]  # Teacher forcing

    else:
        # teacher forchingを使わずデコーダの予測を使用
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  #確率が最大の1語の単語，配列の何番目か
            decoder_input = topi.squeeze().detach()  # detach from history as input

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.item() / target_length


#残り時間とかの見積もり


import time
import math


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


'''
学習のループとlossや時間の表示とか
このplotしてるのはクロスエントロピー誤差のこと？
'''

def trainIters(encoder, decoder, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)
    training_pairs = [tensorsFromPair(random.choice(pairs))
                      for i in range(n_iters)]
    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):
        training_pair = training_pairs[iter - 1]
        input_tensor = training_pair[0]
        target_tensor = training_pair[1]

        loss = train(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))

        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)



#結果のプロット


import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig('loss_'+today_str+'.png')


'''
評価

評価は殆ど訓練と同じですが、ターゲットがないので各ステップについてデコーダの予測を自身に戻して供給します。それが単語を予測するたびにそれを出力文字列に追加してそして EOS トークンを予測する場合にそこで停止します。デコーダの attention 出力もまた後で表示するためにストアします。

訓練セットからのランダムなセンテンスを評価して何某かの主観的な質の判断を行なうために入力、ターゲット、そして出力をプリントアウトすることができます :
'''

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]


def evaluateRandomly(encoder, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = evaluate(encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')




hidden_dim = 256
encoder1 = EncoderRNN(input_lang.n_words, hidden_dim).to(device)
attn_decoder1 = AttnDecoderRNN(hidden_dim, output_lang.n_words, dropout_p=0.1).to(device)

trainIters(encoder1, attn_decoder1, 75000, print_every=5000)
#↑いわゆるlossは(n_iters=75000)*(earning_rate=0.01)のことっぽい

#ここで学習完了



evaluateRandomly(encoder1, attn_decoder1)

'''
↑
学習データからランダムに10個，予測みたいな


> quel imbecile tu es !
= you re such an idiot !
< you re such an idiot ! <eos>

こいうの10個

'''

'''
attentionの可視化の，シンプルな例
引数かえれば他の入力もできる

'''

output_words, attentions = evaluate(
    encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())


'''
↓こっちの方が，軸のラベルとかいろいろついてるやつ

'''

def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_sentence.split(' ') +
                       ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


def evaluateAndShowAttention(input_sentence):
    output_words, attentions = evaluate(
        encoder1, attn_decoder1, input_sentence)
    print('input =', input_sentence)
    print('output =', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)



#↓いろいろ可視化の例
evaluateAndShowAttention("elle a cinq ans de moins que moi .")

evaluateAndShowAttention("elle est trop petit .")

evaluateAndShowAttention("je ne crains pas de mourir .")

evaluateAndShowAttention("c est un jeune directeur plein de talent .")




