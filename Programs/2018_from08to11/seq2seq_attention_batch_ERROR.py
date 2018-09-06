# -*- coding: utf-8 -*-

'''
pytorchのseq2seqチュートリアルを改変
ミニバッチ学習をするため、↓このサイトを参考にしてたが理解不足で実装できていない
http://takoroy-ai.hatenadiary.jp/entry/2018/07/02/224216
おそらくテンソルの次元とかその調整あたりが原因？
### 注意！エラー残ったままで正常に動かない ###
このままではダメというメモ的な

入力データファイルは
xxx-yyy.txt
xxxが翻訳前，yyy翻後の言語

動かしていたバージョン
python   : 2.7.12
pythorch : 2.0.4


#TODO まだここ編集途中
プログラム全体の構成
    ・グローバル変数一覧
    ・関数群
    ・main部

プログラム全体の流れ
    0.いろいろ前準備
    1.学習データの前処理
    2.fasttextのロードと辞書の作成
    3.モデルの定義
    4.モデルの学習
    5.val_loss最小モデルのロード
    6.テストの実行
    7.結果まとめの出力
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

#TODO
#いわゆるmain部的な整理を合同ゼミ後
#タブをスペースに置換



#----- グローバル変数一覧 -----
MAX_LENGTH = 40
hidden_dim = 256

#自分で定義したグローバル関数とか
file_path='./pytorch_data/'
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


#TODO 入出力同じ語彙で管理？
#TODO 語彙はあらかじめ与える？

#seq2seqモデルで用いる語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": 3}
        self.word2count = {"<UNK>": 0}
        self.index2word = {{0: "PAD", 1: "SOS", 2: "EOS", 3: "<UNK>"}
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






#次は学習データの準備



###########################
# 3.入力データ変換
###########################


#単語列をID列に
def indexesFromSentence(lang, sentence):
    return [lang.check_word2index(word) for word in sentence.split(' ')]

#単語列からモデルの入力へのテンソルに
def tensorFromSentence(lang, sentence):
    sent_list=sentence.split(' ')
    length= len(sent_list)
    indexes = [lang.check_word2index(word) for word in sent_list]
    return indexes + [EOS_token] + [0] * (MAX_LENGTH - length - 1), length + 1

'''
#入力と出力のペアからテンソルに
def tensorsFromPair(lang, pair):
    input_tensor = tensorFromSentence(lang, pair[0])
    target_tensor = tensorFromSentence(lang, pair[1])
    return (input_tensor, target_tensor)
'''


#引用0
def generate_batch(pairs, batch_size=200, shuffle=True):
    if shuffle:
        random.shuffle(pairs)
    
    for i in range(len(pairs) // batch_size):
        batch_pairs = pairs[batch_size*i:batch_size*(i+1)]

        input_batch = []
        target_batch = []
        input_lens = []
        target_lens = []
        for input_seq, target_seq in batch_pairs:
            input_seq, input_length = tensorFromSentence(input_lang, input_seq)
            target_seq, target_length = tensorFromSentence(output_lang, target_seq)

            input_batch.append(input_seq)
            target_batch.append(target_seq)
            input_lens.append(input_length)
            target_lens.append(target_length)

        input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
        target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
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
    
    decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)  # (1, b)
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
def batch_evaluation(input_batch, input_lens, target_batch, target_lens, encoder, decoder, criterion):
    with torch.no_grad():
        
        batch_size = input_batch.shape[1]
        target_length = target_lens.max().item()
        target_batch = target_batch[:target_length]

        loss = 0
        
        encoder_outputs, encoder_hidden = encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))
        decoder_input = torch.tensor([SOS_token] * batch_size, device=device)  # (b)
        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        decoded_outputs = torch.zeros(target_length, batch_size, output_lang.n_words, device=device)
        decoded_words = torch.zeros(batch_size, target_length, device=device)
        
        for di in range(target_length):
            decoder_output, decoder_hidden, _ = decoder(
                decoder_input, decoder_hidden, encoder_outputs)  # (b,odim), ((b,h),(b,h)), (b,il)        
            decoded_outputs[di] = decoder_output
            
            loss += criterion(decoder_output, target_batch[di])
        
            _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)
            decoded_words[:, di] = topi[:, 0]  # (b)
            decoder_input = topi.squeeze(1)
        
        bleu = 0
        for bi in range(batch_size):
            try:
                end_idx = decoded_words[bi, :].tolist().index(EOS_token)
            except:
                end_idx = target_length
            score = compute_bleu(
                [[[output_lang.index2word[i] for i in target_batch[:, bi].tolist() if i > 2]]],
                [[output_lang.index2word[j] for j in decoded_words[bi, :].tolist()[:end_idx]]]
            )
            bleu += score

        return loss.item() / target_length, bleu / float(batch_size)
















###########################
# 4.モデルの学習
###########################

'''
モデルの訓練

“Teacher forcing” は(seq2seqのでの)次の入力としてデコーダの推測を使用する代わりに、実際のターゲット出力を各次の入力として使用する概念です。

PyTorch autograd が与えてくれる自由度ゆえに、単純な if ステートメントで "teacher forcing" を使用するか否かをランダムに選択することができます。それを更に使用するためには teacher_forcing_ratio を上向きに調整してください。
'''

'''
学習1回分のクラス

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

    #↑ではlossを全入力に対する和で計算してるので割って平均lossを返す
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
def trainIters(lang, encoder, decoder, pairs, n_iters, print_every=1000, plot_every=100, learning_rate=0.01):
    print("Training...")
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(decoder.parameters(), lr=learning_rate)

    criterion = nn.NLLLoss(ignore_index=PAD_token)

    for iter in range(1, n_iters + 1):
        total_loss = 0
        #学習1データ1回分？
        
        for input_batch, input_lens, target_batch, target_lens in generate_batch(train_pairs, batch_size=batch_size):
            loss = batch_train(input_batch, input_lens, target_batch, target_lens, encoder,
                        decoder, optimizer, criterion, teacher_forcing)
            total_loss += loss
            train_loss = total_loss / (len(train_pairs) / batch_size)
        
        
        print_loss_total += train_loss
        plot_loss_total += train_loss

        #画面にlossと時間表示
        #経過時間 (- 残り時間) (現在のiter 進行度) loss
        if iter % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                         iter, iter / n_iters * 100, print_loss_avg))
        #lossグラフ記録
        if iter % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    #lossグラフ描画
    showPlot(plot_losses)


#グラフの描画（画像ファイル保存）
def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)
    plt.savefig(save_path+'_loss.png')



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
            #TODO ここのdrtachの意味

        #返り値は予測した単語列とattentionの重み？
        return decoded_words, decoder_attentions[:di + 1]

#ランダムにn個のデータ予測
def evaluateRandomly(lang, encoder, decoder, n=3):
    for i in range(n):
        pair = random.choice(pairs)
        print('cloze:', pair[0])
        print('ans  :', pair[1])
        output_words, attentions = evaluate(lang, encoder, decoder, pair[0])
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
    output_words, attentions = evaluate(
        lang, encoder, decoder, input_sentence)
    print('input  :', input_sentence)
    print('output :', ' '.join(output_words))
    showAttention(input_sentence, output_words, attentions)





#----- main部 -----
if __name__ == '__main__':
    # 1.データ読み込み
    #TODO まだ途中
    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)

    cloze_path=file_path+'tmp_cloze.txt'
    ans_path=file_path+'tmp_ans.txt'

    pairs=readData(cloze_path, ans_path)

    # 2.モデル定義
    my_encoder = EncoderRNN(vocab.n_words, hidden_dim).to(my_device)
    my_decoder = AttnDecoderRNN(hidden_dim, vocab.n_words, dropout_p=0.1).to(my_device)


    # 3.学習
    trainIters(vocab, my_encoder, my_decoder, pairs, n_iters=300, print_every=100, plot_every=100)
    #↑lossグラフの横軸は n_iters / plot_every


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
