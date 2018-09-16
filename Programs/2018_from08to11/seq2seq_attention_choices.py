# -*- coding: utf-8 -*-

'''
pytorchのseq2seqチュートリアルを改変
seq2seq_attention.py から変更

空所内のみの予測時に選択肢の利用を追加

ミニバッチ学習は未実装のまま

動かしていたバージョン
python   : 3.5.2
pythorch : 2.0.4

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



#----- グローバル変数一覧 -----
MAX_LENGTH = 40
hidden_dim = 256

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + '/' + today_str
SOS_token = 0
EOS_token = 1
UNK_token = 2

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
        self.word2index = {"<UNK>": 2}
        self.word2count = {"<UNK>": 0}
        self.index2word = {0: "SOS", 1: "EOS", 2: "<UNK>"}
        self.n_words = 3  # SOS と EOS と UNK

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
    print("Reading data...")
    pairs=[]
    i=0
    with open(input_file, encoding='utf-8') as input:
        with open(target_file, encoding='utf-8') as target:
            for line1, line2 in zip(input, target):
                i+=1
                pairs.append([normalizeString(line1), normalizeString(line2)])
    print("data: %s" % i)

    return pairs


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
        return torch.zeros(1, 1, self.hidden_dim, device=my_device)

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
        return torch.zeros(1, 1, self.hidden_dim, device=my_device)


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


#入力と出力のペアからテンソルに
def tensorsFromPair(lang, pair):
    input_tensor = tensorFromSentence(lang, pair[0])
    target_tensor = tensorFromSentence(lang, pair[1])
    return (input_tensor, target_tensor)


###########################
# 4.モデルの学習
###########################

'''
モデルの訓練

“Teacher forcing” は(seq2seqのでの)次の入力としてデコーダの推測を使用する代わりに、実際のターゲット出力を各次の入力として使用する概念です。

PyTorch autograd が与えてくれる自由度ゆえに、単純な if ステートメントで "teacher forcing" を使用するか否かをランダムに選択することができます。それを更に使用するためには teacher_forcing_ratio を上向きに調整してください。
'''
#学習1データ分
def train_onedata(input_tensor, target_tensor, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    #入出力の長さを計算
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)

    encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=my_device)

    loss = 0

    #エンコーダの準備
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(
            input_tensor[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0, 0]

    #デコーダの準備
    decoder_input = torch.tensor([[SOS_token]], device=my_device)

    decoder_hidden = encoder_hidden

    #teacher forcingを使用する割合
    teacher_forcing_ratio = 0.5

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
            #TODO このdetach()の処理よく分からない

            loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

    loss.backward()
    #↑lossはdouble型ではなくVariableクラスになっている
    #backwardメソッドを呼ぶことで逆伝搬がスタート，直前のノードに微分値をセット

    #エンコーダおよびデコーダの学習（パラメータの更新）
    encoder_optimizer.step()
    decoder_optimizer.step()

    #出力が可変長なのでlossも1ノードあたりに正規化
    return loss.item() / target_length


#val_lossを算出1データ分、trainやevaluateとほぼ同じ
def valid_onedata(input_tensor, target_tensor, encoder, decoder, criterion, max_length=MAX_LENGTH):
    with torch.no_grad():
        val_loss = 0

        encoder_hidden = encoder.initHidden()
        input_length = input_tensor.size(0)
        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=my_device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=my_device)

        decoder_hidden = encoder_hidden
        target_length = target_tensor.size(0)

        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.topk(1)  #確率が最大の1語の単語，配列の何番目か
            decoder_input = topi.squeeze().detach()  # detach from history as input
            #TODO このdetach()の処理よく分からない

            val_loss += criterion(decoder_output, target_tensor[di])
            if decoder_input.item() == EOS_token:
                break

        #出力が可変長なのでlossも1ノードあたりに正規化
        return val_loss.item() / target_length


#全データに対して学習1回分
def train(training_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion, max_length=MAX_LENGTH):
    random.shuffle(training_pairs)
    loss=0
    tmp=0
    for pair in training_pairs:
        input_tensor = pair[0]
        target_tensor = pair[1]
        tmp+=1
        loss += train_onedata(input_tensor, target_tensor, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)

    return 1.0*loss/tmp


#val_lossを算出
def valid(valid_pairs, encoder, decoder, criterion):
    val_loss=0
    tmp=0
    for pair in valid_pairs:
        input_tensor = pair[0]
        target_tensor = pair[1]
        tmp+=1
        val_loss += valid_onedata(input_tensor, target_tensor, encoder,
                     decoder, criterion)

    return 1.0*val_loss/tmp


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

    best_val_loss=100   #仮
    best_iter=0

    best_encoder_weight = copy.deepcopy(encoder.state_dict())
    best_decoder_weight = copy.deepcopy(decoder.state_dict())

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    training_pairs = [tensorsFromPair(lang, p) for p in train_pairs]
    valid_pairs = [tensorsFromPair(lang, p) for p in val_pairs]

    criterion = nn.NLLLoss()

    for iter in range(1, n_iters + 1):

        loss = train(training_pairs, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)

        val_loss = valid(valid_pairs, encoder, decoder, criterion)

        print_loss_total += loss
        plot_loss_total += loss

        print_val_loss_total += val_loss
        plot_val_loss_total += val_loss

        #画面にlossと時間表示
        #経過時間 (- 残り時間) (現在のiter 進行度) loss val_loss
        if iter == 1:
            print('%s (%d %d%%) loss=%.4f, val_loss=%.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_total, print_val_loss_total))

        elif iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print_val_loss_avg = print_val_loss_total / print_every
            print_val_loss_total = 0
            print('%s (%d %d%%) loss=%.4f, val_loss=%.4f' % (timeSince(start, iter / n_iters), iter, iter / n_iters * 100, print_loss_avg, print_val_loss_avg))

        #lossグラフ記録
        plot_loss_avg = plot_loss_total
        plot_losses.append(plot_loss_avg)
        plot_loss_total = 0

        plot_val_loss_avg = plot_val_loss_total
        plot_val_losses.append(plot_val_loss_avg)
        plot_val_loss_total = 0

        #val_loss最小更新
        if (best_val_loss > val_loss) or (iter == 1):
            best_val_loss = val_loss
            best_iter=iter
            best_encoder_weight = copy.deepcopy(encoder.state_dict())
            best_decoder_weight = copy.deepcopy(decoder.state_dict())


    #lossグラフ描画
    showPlot2(plot_losses, plot_val_losses)
    #oldshowPlot(plot_losses)
    #TODO val_lossの描画も

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
        #以下はほぼtrainと同じ
        input_tensor = tensorFromSentence(lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=my_device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=my_device)  # SOS

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
                #EOSならば終了
                break
            else:
                decoded_words.append(lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()
            #TODO ここのdetachの意味

        #返り値は予測した単語列とattentionの重み？
        return decoded_words, decoder_attentions[:di + 1]



'''
空所の予測方法2つある？
1（共通）.空所が始まるまで（「{」のところまでは正答文そのまま、「{」から予測スタート）
2a.「}」が出るまでデコーダに予測させる
2b.答えを見て空所内の単語数は固定

ひとまず2aの方でevaluate_cloze()は実装してる
'''
#空所内のみを予想
#evaluate()の拡張
def evaluate_cloze(lang, encoder, decoder, input_sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        #no_grad()の間はパラメータが固定される（更新されない）
        input_tensor = tensorFromSentence(lang, input_sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=my_device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=my_device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        tmp_list=normalizeString(input_sentence).split(' ')
        tmp_list.append('<EOS>')
        cloze_start=tmp_list.index('{')
        cloze_end=tmp_list.index('}')
        cloze_flag=0
        cloze_words=0

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            #空所が始まるまでは空所外の部分はそのまま用いる
            #ここではEOSを考慮しなくてよい
            if di <= cloze_start:
                decoded_words.append(tmp_list[di])
                decoder_input = input_tensor[di]

            #空所内の予測
            elif cloze_flag == 0:
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    #EOSならば終了
                    break
                else:
                    word=lang.index2word[topi.item()]
                    decoded_words.append(word)
                    decoder_input = topi.squeeze().detach()
                    if word == '}':
                        cloze_flag=1
                    else:
                        cloze_words+=1

            #空所後の予測
            else:
                word=tmp_list[di-cloze_words]
                decoded_words.append(word)
                if word == '<EOS>':
                    break
                else:
                    decoder_input = input_tensor[di-cloze_words]

        #返り値は予測した単語列とattentionの重み？
        return decoded_words, decoder_attentions[:di + 1]


#空所内のみを予想かつ選択肢の利用
#evaluate_clozeの拡張
def evaluate_choice(lang, encoder, decoder, input_sentence, choices, max_length=MAX_LENGTH):
    with torch.no_grad():
        #no_grad()の間はパラメータが固定される（更新されない）
        input_tensor = tensorFromSentence(lang, input_sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_dim, device=my_device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(
                input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=my_device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        tmp_list=normalizeString(input_sentence).split(' ')
        tmp_list.append('<EOS>')
        cloze_start=tmp_list.index('{')
        cloze_end=tmp_list.index('}')
        cloze_flag=0
        cloze_words=0
        choice_flag=0

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data

            #空所が始まるまでは空所外の部分はそのまま用いる
            #ここではEOSを考慮しなくてよい
            if di <= cloze_start:
                decoded_words.append(tmp_list[di])
                decoder_input = input_tensor[di]

            #空所内の予測
            #TODO ここ変更、選択肢から選ぶ

            elif cloze_flag == 0:
                '''
                #TODO
                    １:選択肢4つを読み込み
                    2:４つの先頭1語を見て、どの1語が最も確率高いか
                    3:もし2で選んだ選択肢が2語以上、かつ他の選択肢との共通部分あるなら再度似たような処理

                    選択肢4つの語の被りを見るchoice_flagが必要？
                    まず重複を見てから？


                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    #EOSならば終了
                    break
                else:
                    word=lang.index2word[topi.item()]
                    decoded_words.append(word)
                    decoder_input = topi.squeeze().detach()
                    if word == '}':
                        cloze_flag=1
                    else:
                        cloze_words+=1
                '''

            #空所後の予測
            else:
                word=tmp_list[di-cloze_words]
                decoded_words.append(word)
                if word == '<EOS>':
                    break
                else:
                    decoder_input = input_tensor[di-cloze_words]

        #返り値は予測した単語列とattentionの重み？
        return decoded_words, decoder_attentions[:di + 1]


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
        if pred == ans:
            allOK+=1
        pred_cloze = get_cloze(pred)
        ans_cloze = get_cloze(ans)
        tmp_ans_length=len(ans_cloze.split(' '))
        line_num+=1
        if is_correct_cloze(pred):
            tmp_match=match(pred_cloze, ans_cloze)
            if tmp_match > 0:
                partOK+=1
            if tmp_ans_length == tmp_match:
                clozeOK+=1
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
            line=get_cloze(normalizeString(line))
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


def score(preds, ans, file_output, file_name):
    #精度のprintとファイル出力
    line, allOK, clozeOK, partOK, BLEU, miss = calc_score(preds, ans)
    #TODO 今は実装してないが必要に応じてchange_unkの精度計算も作る？
    print_score(line, allOK, clozeOK, partOK, BLEU, miss) #仮の関数名
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
        output_cloze_words, cloze_attentions = evaluate_cloze(lang, encoder, decoder, input_sentence)
        preds_cloze.append(' '.join(output_cloze_words))
        #output_choice_words, choice_attentions = evaluate_choice(lang, encoder, decoder, input_sentence, choi)
        #preds_choices.append(' '.join(output_choice_words))

        if saveAttention:
            showAttention('all', input_sentence, output_words, attentions)
            showAttention('cloze', input_sentence, output_cloze_words, cloze_attentions)
            #showAttention('choice', input_sentence, output_choice_words, choice_attentions)
        if file_output:
            output_preds(save_path+'preds.txt', preds)
            output_preds(save_path+'preds_cloze.txt', preds_cloze)
            #output_preds(save_path+'preds_choices.txt', preds_choices)
    print("Calc scores ...")
    score(preds, ans, file_output, save_path+'score.txt')
    score(preds_cloze, ans, file_output, save_path+'score_cloze.txt')
    #score(preds_choices, ans, file_output, save_path+'score_choices.txt')


#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--encoder', help='encoder file name (when load model, mode=test)')
    parser.add_argument('--decoder', help='decoder file name (when load model, mode=test)')
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
    my_encoder = EncoderRNN(vocab.n_words, hidden_dim).to(my_device)
    my_decoder = AttnDecoderRNN(hidden_dim, vocab.n_words, dropout_p=0.1).to(my_device)

    #学習時
    if args.mode == 'all' or args.mode == 'mini':
        train_cloze=file_path+'tmp_cloze.txt'
        train_ans=file_path+'tmp_ans.txt'

        all_data=readData(train_cloze, train_ans)
        if args.mode == 'mini':
            all_data=all_data[:20]

        train_data, val_data = train_test_split(all_data, test_size=0.1)

        #モデルとか結果とかを格納するディレクトリの作成
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'

        # 3.学習
        my_encoder, my_decoder = trainIters(vocab, my_encoder, my_decoder, train_data, val_data, n_iters=3, saveModel=True)

    #すでにあるモデルでテスト時
    else:
        save_path=args.model_dir+'/'

        my_encoder.load_state_dict(torch.load(save_path+args.encoder))
        my_decoder.load_state_dict(torch.load(save_path+args.decoder))

        save_path=save_path+today_str

    # 4.評価
    test_cloze=file_path+'center_cloze.txt'
    test_ans=file_path+'center_ans.txt'
    test_choi=file_path+'center_choices.txt'

    test_data=readData(test_cloze, test_ans)
    choices=get_choices(test_choi)
    #choices=['', '', '', '', '', '']
    if args.mode == 'mini' or args.mode == 'mini_test':
        test_data=test_data[:5]
        choices=choices[:5]

    #テストデータに対する予測と精度の計算
    #選択肢を使ったテスト
    test_choices(vocab, my_encoder, my_decoder, test_data, choices, saveAttention=False, file_output=True)
