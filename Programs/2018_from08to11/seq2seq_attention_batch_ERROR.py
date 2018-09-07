# -*- coding: utf-8 -*-

'''
### 注意！エラー残ったままで正常に動かない ###
このままではダメというメモ的な

pytorchのseq2seqチュートリアルを改変
seq2seq_attention_with_vocab.py　から変更
ミニバッチ学習をするため、↓このサイトを参考に実装しようとしたが理解不足で実装できていない
http://takoroy-ai.hatenadiary.jp/entry/2018/07/02/224216

File "seq2seq_attention_batch_retry.py", line 380, in forward
    drop_encoder_outputs.transpose(0, 1)  # (b, ml, 2h)
RuntimeError: invalid argument 2: wrong matrix size, batch1: 1x20, batch2: 24x512 at /pytorch/aten/src/TH/generic/THTensorMath.c:2275


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





#----- グローバル変数一覧 -----
MAX_LENGTH = 40
hidden_dim = 256

emb_size = 100
hidden_size = 256
attn_size = 128



#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str
PAD_token = 0
SOS_token = 1
EOS_token = 2
UNK_token = 3


#事前処理いろいろ
print('Start: '+today_str)
my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#----- 関数群 -----


###########################
# 1.データの準備
###########################


#seq2seqモデルで用いる語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": 3}
        self.word2count = {"<UNK>": 0}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS", 3: "<UNK>"}
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
def normalizeString(s):
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
    s = re.sub(r'[^a-z{}]', ' ', s)
    s = re.sub(r'[ ]+', ' ', s)

    return s

#与えた語彙読み込み(自作)
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(normalizeString(line))
    print("Vocab: %s" % lang.n_words)

    return lang

#入出力データ読み込み用
def readData(input_file, target_file):
    print("Reading train data...")
    pairs=[]
    i=0
    with open(input_file, encoding='utf-8') as input:
        with open(target_file, encoding='utf-8') as target:
            for line1, line2 in zip(input, target):
                i+=1
                pairs.append([normalizeString(line1), normalizeString(line2)])
    print("Train data: %s" % i)

    return pairs







###########################
# 2.モデル定義
###########################

#クラスのこととかよく分かってないのでコメント過剰気味



#エンコーダのクラス
#nn.Moduleクラスを継承
#これはチュートリアルのやつそのまま
class oldEncoderRNN(nn.Module):
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
        return torch.zeros(1, 1, self.hidden_dim, device=my_device)

'''
もしembeddingの初期値与えたかったら

embed = nn.Embedding(num_embeddings, embedding_dim)
# pretrained_weight is a numpy matrix of shape (num_embeddings, embedding_dim)
embed.weight.data.copy_(torch.from_numpy(pretrained_weight))
'''



#attentionつきデコーダのクラス
#これはチュートリアルのやつそのまま
class oldAttnDecoderRNN(nn.Module):
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
        return torch.zeros(1, 1, self.hidden_dim, device=my_device)



#以下引用したエンコーダおよびデコーダに加筆
class EncoderRNN(nn.Module):
    def __init__(self, input_size, emb_size, hid_size, pad_token=0):
        super(EncoderRNN, self).__init__()
        self.embedding_size = emb_size
        self.hidden_size = hid_size

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hid_size,
                            bidirectional=True)
        self.linear_h = nn.Linear(hid_size * 2, hid_size)
        self.linear_c = nn.Linear(hid_size * 2, hid_size)
        #チュートリアルではGRUだったものをLSTMに変更
        #LSTMなので入力はinputとhiddenで2つ?、出力はoutputと(hidden_h, hidden_c)で2つ
        #さらにbidirectionalなので出力は次元数が2倍になる
        #TODO pytorchの公式ドキュメント見る

    def forward(self, input_batch, input_lens):
        """
        :param input_batch: (s, b)
        :param input_lens: (b)

        :returns (s, b, 2h), ((1, b, h), (1, b, h))
        s:英文の長さ
        b:バッチサイズ
        h:隠れ層の次元数、2hはその2倍
        """

        batch_size = input_batch.shape[1]
        #ここbatch_size =input_lenでいいのでは？

        embedded = self.embedding(input_batch)  # (s, b) -> (s, b, h)
        output, (hidden_h, hidden_c) = self.lstm(embedded)

        hidden_h = hidden_h.transpose(1, 0)  # (2, b, h) -> (b, 2, h)
        hidden_h = hidden_h.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_h = F.dropout(hidden_h, p=0.5, training=self.training)
        hidden_h = self.linear_h(hidden_h)  # (b, 2h) -> (b, h)
        hidden_h = F.relu(hidden_h)
        hidden_h = hidden_h.unsqueeze(0)  # (b, h) -> (1, b, h)

        hidden_c = hidden_c.transpose(1, 0)
        hidden_c = hidden_c.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_c = F.dropout(hidden_c, p=0.5, training=self.training)
        hidden_c = self.linear_c(hidden_c)
        hidden_c = F.relu(hidden_c)
        hidden_c = hidden_c.unsqueeze(0)  # (b, h) -> (1, b, h)

        return output, (hidden_h, hidden_c)  # (s, b, 2h), ((1, b, h), (1, b, h))


class DecoderRNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_token=0):
        super(DecoderRNN, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTMCell(emb_size, hidden_size)
        self.out_w = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        :param input: (b)
        :param hidden: ((b,h), (b,h))
        :return: (b,o), (b,h)
        """

        embedded = self.embedding(input)  # (b) -> (b,e)
        decoder_output, hidden = self.lstm(embedded, hidden)  # (b,e),((b,h),(b,h)) -> (b,h),((b,h),(b,h))
        output = self.out_w(decoder_output)  # (b,h) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, hidden  # (b,o), (b,h)


class AttnDecoderRNN1(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, max_length=20):
        super(AttnDecoderRNN1, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, emb_size)
        self.attn = nn.Linear(emb_size+2*hidden_size, max_length)
        self.attn_combine = nn.Linear(emb_size+2*hidden_size, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, encoder_outputs):
        """
        :param input: (b)
        :param hidden: ((b, h), (b, h))
        :param encoder_outputs: (il, b, 2h)
        :return: (b,o), ((b,h),(b,h)), (b,il)
        """
        input_length = encoder_outputs.shape[0]
        #padding
        encoder_outputs = torch.cat([
            encoder_outputs,
            torch.zeros(
                self.max_length - input_length,
                encoder_outputs.shape[1],
                encoder_outputs.shape[2],
                device=my_device
            )
        ], dim=0)  # (il,b,2h), (ml-il,b,2h) -> (ml,b,2h)
        drop_encoder_outputs = F.dropout(encoder_outputs, p=0.1, training=self.training)

        # embedding
        embedded = self.embedding(input)  # (b) -> (b,e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)

        emb_hidden = torch.cat([embedded, hidden[0], hidden[1]], dim=1)  # (b,e),((b,h),(b,h)) -> (b,e+2h)

        attn_weights = self.attn(emb_hidden)  # (b,e+2h) -> (b,ml)
        attn_weights = F.softmax(attn_weights, dim=1)

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),  # (b, 1, ml)
            drop_encoder_outputs.transpose(0, 1)  # (b, ml, 2h)
        )  # -> (b, 1, 2h)

        attn_applied = F.dropout(attn_applied, p=0.1, training=self.training)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # ((b,e),(b,2h)) -> (b,e+2h)
        output = self.attn_combine(output)  # (b,e+2h) -> (b,h)
        output = F.dropout(output, p=0.5, training=self.training)

        output = F.relu(output)
        hidden = self.lstm(output, hidden)  # (b,h),((b,h),(b,h)) -> (b,h)((b,h),(b,h))

        output = F.log_softmax(self.out(hidden[0]), dim=1)  # (b,h) -> (b,o)
        return output, hidden, attn_weights  # (b,o),(b,h),(b,il)


class AttnDecoderRNN2(nn.Module):
    def __init__(self, emb_size, hidden_size, attn_size, output_size, pad_token=0):
        super(AttnDecoderRNN2, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTMCell(emb_size, hidden_size)

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

        hidden = self.lstm(embedded, hidden)  # (b,e),((b,h),(b,h)) -> ((b,h),(b,h))
        decoder_output = torch.cat(hidden, dim=1)  # ((b,h),(b,h)) -> (b,2h)
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
        attentional = F.tanh(attentional)
        #attentional = F.dropout(attentional, p=0.5, training=self.training)

        output = self.out_w(attentional)  # (b,a) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights.squeeze(2)  # (b,o), ((b,h),(b,h)), (b,il)







#次は学習データの準備

###########################
# 3.入力データ変換
###########################

#単語列からモデルの入力へのテンソルに
def tensorFromSentence(lang, sentence):
    sent_list=sentence.split(' ')
    length= len(sent_list)
    indexes = [lang.check_word2index(word) for word in sent_list]
    return indexes + [EOS_token] + [0] * (MAX_LENGTH - length - 1), length + 1


#引用0
def generate_batch(lang, pairs, batch_size=200, shuffle=True):
    if shuffle:
        random.shuffle(pairs)

    for i in range(len(pairs) // batch_size):
        batch_pairs = pairs[batch_size*i:batch_size*(i+1)]

        input_batch = []
        target_batch = []
        input_lens = []
        target_lens = []
        for input_seq, target_seq in batch_pairs:
            input_seq, input_length = tensorFromSentence(lang, input_seq)
            target_seq, target_length = tensorFromSentence(lang, target_seq)

            input_batch.append(input_seq)
            target_batch.append(target_seq)
            input_lens.append(input_length)
            target_lens.append(target_length)

        input_batch = torch.tensor(input_batch, dtype=torch.long, device=my_device)
        target_batch = torch.tensor(target_batch, dtype=torch.long, device=my_device)
        input_lens = torch.tensor(input_lens)
        target_lens = torch.tensor(target_lens)

        # sort
        input_lens, sorted_idxs = input_lens.sort(0, descending=True)
        input_batch = input_batch[sorted_idxs].transpose(0, 1)
        input_batch = input_batch[:input_lens.max().item()]

        target_batch = target_batch[sorted_idxs].transpose(0, 1)
        target_batch = target_batch[:target_lens.max().item()]
        target_lens = target_lens[sorted_idxs]

        yield input_batch, input_lens, target_batch, target_lens



#引用1
def batch_train(input_batch, input_lens, target_batch, target_lens,
                encoder, decoder, optimizer, criterion,
                teacher_forcing_ratio=0.5):

    loss = 0
    optimizer.zero_grad()

    batch_size = input_batch.shape[1]
    target_length = target_lens.max().item()

    encoder_outputs, encoder_hidden = encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))

    decoder_input = torch.tensor([[SOS_token] * batch_size], device=my_device)  # (1, b)
    decoder_inputs = torch.cat([decoder_input, target_batch], dim=0)  # (1,b), (n,b) -> (n+1, b)
    decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input

        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_inputs[di], decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, decoder_inputs[di+1])
    else:
        decoder_input = decoder_inputs[0]
        for di in range(target_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            loss += criterion(decoder_output, decoder_inputs[di+1])

            _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)
            decoder_input = topi.squeeze(1).detach()

    loss.backward()

    optimizer.step()

    return loss.item() / target_length


#引用２
#一部変更
def batch_evaluation(lang, input_batch, input_lens, target_batch, target_lens, encoder, decoder, criterion):
    with torch.no_grad():

        batch_size = input_batch.shape[1]
        target_length = target_lens.max().item()
        target_batch = target_batch[:target_length]

        loss = 0

        encoder_outputs, encoder_hidden = encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))
        decoder_input = torch.tensor([SOS_token] * batch_size, device=my_device)  # (b)
        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        decoded_outputs = torch.zeros(target_length, batch_size, lang.n_words, device=my_device)
        decoded_words = torch.zeros(batch_size, target_length, device=my_device)

        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)  # (b,odim), ((b,h),(b,h)), (b,il)
            decoded_outputs[di] = decoder_output

            loss += criterion(decoder_output, target_batch[di])

            _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)
            decoded_words[:, di] = topi[:, 0]  # (b)
            decoder_input = topi.squeeze(1)
        '''
        bleu = 0
        for bi in range(batch_size):
            try:
                end_idx = decoded_words[bi, :].tolist().index(EOS_token)
            except:
                end_idx = target_length
            score = compute_bleu(
                [[[lang.index2word[i] for i in target_batch[:, bi].tolist() if i > 2]]],
                [[lang.index2word[j] for j in decoded_words[bi, :].tolist()[:end_idx]]]
            )
            bleu += score
        '''

        #return loss.item() / target_length, bleu / float(batch_size)
        return loss.item() / target_length






###########################
# 4.モデルの学習
###########################

'''
モデルの訓練

“Teacher forcing” は(seq2seqのでの)次の入力としてデコーダの推測を使用する代わりに、実際のターゲット出力を各次の入力として使用する概念です。

PyTorch autograd が与えてくれる自由度ゆえに、単純な if ステートメントで "teacher forcing" を使用するか否かをランダムに選択することができます。それを更に使用するためには teacher_forcing_ratio を上向きに調整してください。
'''




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


#グラフの描画（画像ファイル保存）
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(save_path+'_loss.png')


#これ追加分
#一部加筆 or 削除
def trainIters(lang, train_pairs, encoder, decoder, n_iter=30, batch_size=200, teacher_forcing=0.5, print_every=10):
    print("Training...")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    optimizer = optim.Adam([p for p in encoder.parameters()] + [p for p in decoder.parameters()])
    criterion = nn.NLLLoss(ignore_index=PAD_token)

    #validation_bleus = []

    for iter in range(1, n_iter + 1):
        total_loss = 0
        #学習
        for input_batch, input_lens, target_batch, target_lens in generate_batch(lang, train_pairs, batch_size=batch_size):
            loss = batch_train(input_batch, input_lens, target_batch, target_lens, encoder,
                        decoder, optimizer, criterion, teacher_forcing)
            total_loss += loss
        train_loss = total_loss / (len(train_pairs) / batch_size)

        print_loss_total += train_loss
        plot_loss_total += train_loss

        '''
        #学習データでの予測精度計算
        for input_batch, input_lens, target_batch, target_lens in generate_batch(train_pairs, batch_size=batch_size, shuffle=False):
            loss, bleu = batch_evaluation(lang, input_batch, input_lens, target_batch, target_lens, encoder, decoder, criterion)
            total_bleu += bleu
        train_bleu = total_bleu / (len(train_pairs) / batch_size)

        total_loss = 0
        total_bleu = 0

        #検証データでの予測精度計算
        for input_batch, input_lens, target_batch, target_lens in generate_batch(lang, test_pairs, batch_size=batch_size, shuffle=False):
            loss, bleu = batch_evaluation(lang, input_batch, input_lens, target_batch, target_lens, encoder, decoder, criterion)
            total_loss += loss
            total_bleu += bleu
        validation_loss = total_loss / (len(test_pairs) / batch_size)
        validation_bleu = total_bleu / (len(test_pairs) / batch_size)
        '''
        #画面にlossと時間表示
        #経過時間 (- 残り時間) (現在のiter 進行度) loss
        if (iter % print_every == 0) or (iter==1):
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iter),
                                         iter, iter / n_iter * 100, print_loss_avg))
        #lossグラフ記録
        plot_loss_avg = plot_loss_total / plot_every
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0
    #lossグラフ描画
    showPlot(plot_losses)

    return train_loss


###########################
# 5.モデルによる予測
###########################
def inference(lang, encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_indxs, input_length = tensorFromSentence(lang, sentence)
        input_batch = torch.tensor([input_indxs], dtype=torch.long, device=my_device)  # (1, s)
        input_length = torch.tensor([input_length])  # (1)

        encoder_outputs, encoder_hidden = encoder(input_batch.transpose(0, 1), input_length)

        decoder_input = torch.tensor([SOS_token], device=my_device)  # (1)

        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))

        decoded_words = []
        attentions = []

        for di in range(max_length):
            decoder_output, decoder_hidden, attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)  # (1,odim), ((1,h),(1,h)), (l,1)
            attentions.append(attention)
            _, topi = decoder_output.topk(1)  # (1, 1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi[0]

        attentions = torch.cat(attentions, dim=0)  # (l, n)

        return decoded_words, attentions.squeeze(0).cpu().numpy()




#ランダムにn個のデータ予測
def evaluateRandomly(lang, train_pairs, encoder, decoder, n=3):
    for i in range(n):
        pair = random.choice(train_pairs)
        print('cloze:', pair[0])
        print('ans  :', pair[1])
        output_words, attentions = inference(lang, encoder, decoder, pair[0])
        output_sentence = ' '.join(output_words)
        print('pred :', output_sentence)
        print('')



#attentionの重みの対応グラフの描画
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
    if len(input_sentence)>10:
        plt.savefig(save_path + input_sentence[:10] + '_attn.png')
    else:
        plt.savefig(save_path + input_sentence + '_attn.png')


def evaluateAndShowAttention(lang, encoder, decoder, input_sentence):
    output_words, attentions = inference(
        lang, encoder, decoder, input_sentence)
    print('input  :', input_sentence)
    print('output :', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)





#----- main部 -----
if __name__ == '__main__':
    # 1.データ読み込み
    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)

    cloze_path=file_path+'tmp_cloze.txt'
    ans_path=file_path+'tmp_ans.txt'

    pairs=readData(cloze_path, ans_path)

    # 2.モデル定義
    my_encoder = EncoderRNN(vocab.n_words, emb_size, hidden_size).to(my_device)
    my_decoder = AttnDecoderRNN1(emb_size, hidden_size, vocab.n_words).to(my_device)



    # 3.学習
    #trainIters(vocab, my_encoder, my_decoder, pairs, n_iter=300, print_every=100, plot_every=100)
    #↑lossグラフの横軸は n_iter / plot_every

    trainIters(vocab, pairs, my_encoder, my_decoder, batch_size=200, n_iter=3, teacher_forcing=0.9)


    # 4.評価
    evaluateRandomly(vocab, my_encoder, my_decoder)




    #↓いろいろ可視化の例
    #センターからいくつか
    evaluateAndShowAttention(vocab, my_encoder, my_decoder, "something s wrong with the car we must have a { } tire")
    #something s wrong with the car we must have a { flat } tire


    '''
    evaluateAndShowAttention(vocab, my_encoder, my_decoder, "taro is now devoting all his time and energy { } english")
    #taro is now devoting all his time and energy { to studying } english

    evaluateAndShowAttention(vocab, my_encoder, my_decoder, "hurry up or we ll be late don t worry i ll be ready { } two minutes")
    #hurry up or we ll be late don t worry i ll be ready { in } two minutes

    evaluateAndShowAttention(vocab, my_encoder, my_decoder, "robin suddenly began to feel nervous { } the interview")
    #robin suddenly began to feel nervous { during } the interview
    '''
    #TODO 正解率の算出とか自分で追加必要
