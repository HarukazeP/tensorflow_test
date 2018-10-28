# -*- coding: utf-8 -*-

'''
pytorchのRNNLMチュートリアルを改変
LSTMを使った言語モデル

動かしていたバージョン
python  : 3.5.2
pytorch : 2.0.4

'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
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

import os
import argparse
import copy

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

#----- グローバル変数一覧 -----

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
git_data_path='../../Data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + '/RNNLM' + today_str

UNK_token = 0

#事前処理いろいろ
print('Start: '+today_str)
if torch.cuda.is_available():
    device = torch.device("cuda")
    print('Use GPU')
else:
    device= torch.device("cpu")

#----- 関数群 -----

#data.py内
class Dictionary:
    def __init__(self):
        self.word2idx = {"<UNK>": UNK_token}
        self.idx2word = {UNK_token: "<UNK>"}
        self.n_words = 1  # UNK

    #文から単語を語彙へ
    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.add_word(word)

    #語彙のカウント
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = self.n_words
            self.idx2word[self.n_words] = word
            self.n_words += 1

    def check_word2idx(self, word):
        if word in self.word2idx:
            return self.word2idx[word]
        else:
            return self.word2idx["<UNK>"]


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
    lang = Dictionary()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(normalizeString(line))
    #print("Vocab: %s" % lang.n_words)

    return lang

#文字列からID列に
def data_tokenize(file, lang):
    all_ids=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=normalizeString(line)
            words = line.split() + ['<eos>']
            for word in words:
                all_ids.append(lang.check_word2idx(word))

    return all_ids

#ID列からデータ作成
def make_data(data, N):
    all_X=[]
    all_Y=[]
    for i in range(len(data)-N):
        all_X.append(data[i:i+N])
        all_Y.append([data[i+N]])

    train_X, val_X = train_test_split(all_X, test_size=0.1)
    train_Y, val_Y = train_test_split(all_Y, test_size=0.1)

    train_X=torch.tensor(train_X, dtype=torch.long, device=device)
    train_Y=torch.tensor(train_Y, dtype=torch.long, device=device)
    val_X=torch.tensor(val_X, dtype=torch.long, device=device)
    val_Y=torch.tensor(val_Y, dtype=torch.long, device=device)

    bsz=args.batch_size
    train_batch = train_X.size(0) // bsz
    train_X = train_X.narrow(0, 0, train_batch * bsz)
    train_Y = train_Y.narrow(0, 0, train_batch * bsz)

    val_batch = val_X.size(0) // bsz
    val_X = val_X.narrow(0, 0, val_batch * bsz)
    val_Y = val_Y.narrow(0, 0, val_batch * bsz)

    train_data = TensorDataset(train_X, train_Y)
    val_data = TensorDataset(val_X, val_Y)

    return train_data, val_data

#model.py内
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, dropout=dropout)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', 'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was supplied,
                                 options are ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, dropout=dropout)
        self.decoder = nn.Linear(nhid*args.ngrams, ntoken) #(入力次元数, 出力次元数)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden):
        emb = self.drop(self.encoder(input))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output) #(文長、バッチサイズ、隠れ層の次元数)
        output = output.transpose(0,1).contiguous() #(バッチサイズ、文長、隠れ層の次元数)
        output = output.view(output.size(0), -1)

        decoded = self.decoder(output)
        return decoded, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)





###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(ntokens, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(args.batch_size)
    loader = DataLoader(data_source, batch_size=args.batch_size, shuffle=False)
    with torch.no_grad():
        for x, y in loader:
            data=x.transpose(0,1)
            targets=y.squeeze()
            output, hidden = model(data, hidden)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
            hidden = repackage_hidden(hidden)
    return total_loss / len(data_source)


def train(ntokens, train_data) :
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0.
    print_loss = 0.
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size)
    loader_train = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    i=0
    batch=0
    batch_set_num=len(train_data)
    for x, y in loader_train:
        '''
        if i==0:
            print(x.size())
            print(y.size())
            i=1
        '''
        batch+=len(x)
        data=x.transpose(0,1)
        #targets=y.transpose(0,1)
        targets=y.squeeze()

        hidden = repackage_hidden(hidden)
        model.zero_grad()
        output, hidden = model(data, hidden)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.item()
        print_loss += loss.item()

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = print_loss / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, batch_set_num, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            print_loss = 0
            start_time = time.time()

    return total_loss/len(train_data)


def showPlot2(loss, val_loss):
    plt.plot(loss, color='blue', marker='o', label='loss')
    plt.plot(val_loss, color='green', marker='o', label='val_loss')
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.savefig(save_path+'loss.png')


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


def get_choices(file_name):
    print("Reading data...")
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=get_cloze(normalizeString(line, choices=True))
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


def get_cloze(line):
    line=re.sub(r'.*{ ', '', line)
    line=re.sub(r' }.*', '', line)

    return line


#選択肢を補充した文4つを返す
def make_sents(choices, cloze_sent):
    sents=[]
    before=re.sub(r'{.*', '', cloze_sent)
    after=re.sub(r'.*}', '', cloze_sent)
    for choice in choices:
        tmp=before + choice + after
        sents.append(tmp.strip())

    return sents


#ファイルから選択肢補充済み文と答えのセット
def make_data_for_sent_score(data_pair, choices_lists, one_word=True):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        if(one_word==True):
            for choice in choices:
                if(len(choice.split(' '))>1):
                    flag=-1
                    #選択肢に2語以上のものがあるときはflagが負
                if(flag>0):
                    test_data=make_sents(choices, sent[0])
                    test_data.append(sent[1])
                    data.append(test_data)
        else:
            test_data=make_sents(choices, sent[0])
            test_data.append(sent[1])
            data.append(test_data)

    return data


#ファイルから選択肢補充済み文と答えのセット
#全単語から
def make_data_for_sent_score_from_all_words(data_pair, choices_lists, all_words):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        for choice in choices:
            if(len(choice.split(' '))>1):
                flag=-1
                #選択肢に2語以上のものがあるときはflagが負
        if(flag>0):
            test_data=make_sents(all_words, sent[0])
            test_data.append(sent[1])
            data.append(test_data)

    return data


def print_score(line, OK):
    print('  acc: ', '{0:.2f}'.format(1.0*OK/line*100),' %')
    print(' line: ',line)
    print('   OK: ',OK)

#正答率の算出
def calc_acc(lang, data, model, N):
    line=0
    OK=0
    for one_data in data:
        line+=1
        ans=one_data[-1]
        ans=ans.replace('{ ', '')
        ans=ans.replace(' }', '')
        ans.strip()
        pred=get_best_sent(lang, one_data[:len(one_data)-1], model, N)
        if pred == ans:
            OK+=1
    print_score(line, OK)


#1文 → ngramのpair
#例えば3-gramなら
#return [[[w1, w2, w3], w4], [[w2, w3, w4], w5], ...]
#文の長さ<Nとなることはない前提
def sent_to_ngram_pair(sent, N):
    pair=[]
    words=sent.split(' ')
    for i in range(len(words)-N-1):
        one_pair=[]
        one_pair.append(words[0+i:N+i])
        one_pair.append(words[N+i])
        pair.append(one_pair)

    return pair

def sent_to_idxs(sent, lang):
    idxs=[]
    for word in sent:
        idxs.append(lang.check_word2idx(word))

    return idxs


#TODO まだ途中
#ngramのペアからモデルの返す尤度をもとにスコアを算出
def calc_sent_score(lang, ngram_pair, model):
    score=0
    n_gram=args.ngrams
    batch=1
    hidden = model.init_hidden(batch)
    with torch.no_grad():
        for one_pair in ngram_pair:
            ids=sent_to_idxs(one_pair[0], lang)
            input = torch.tensor(ids, dtype=torch.long).to(device)
            input = input.unsqueeze(0)  #(1, N)
            input.transpose(0,1)
            output, _ = model(input, hidden)    #(1, 語彙数)
            probs=F.log_softmax(output.squeeze())
            word_idx=lang.check_word2idx(one_pair[1])
            score+=probs[word_idx].item()

    #返り値のスコアは文長で正規化する
    return score/len(ngram_pair)


#1つの問題に対する，選択肢補充済み文複数から
#ベスト1文を返す
def get_best_sent(lang, sents, model, N):
    best_score = -1000.0 #仮
    i=0
    #TODO モデルの返り値は尤度？対数尤度？
    #それによってbest_scoreの初期値変わる
    best_sent=''
    for sent in sents:
        ngram_pair=sent_to_ngram_pair(sent, N)
        #scoreは -inf ～ 0
        score=calc_sent_score(lang, ngram_pair, model)
        if(score<best_score or i==0):
            best_score=score
            best_sent=sent
        i+=1
    return best_sent


#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='LSTM',
                        help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
    parser.add_argument('--emsize', type=int, default=200,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=200,
                        help='number of hidden units per layer')
    parser.add_argument('--nlayers', type=int, default=2,
                        help='number of layers')
    parser.add_argument('--lr', type=float, default=20,
                        help='initial learning rate')
    parser.add_argument('--clip', type=float, default=0.25,
                        help='gradient clipping')
    parser.add_argument('--epochs', type=int, default=30,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=128, metavar='N',
                        help='batch size')
    parser.add_argument('--bptt', type=int, default=35,
                        help='sequence length')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='dropout applied to layers (0 = no dropout)')
    parser.add_argument('--tied', action='store_true',
                        help='tie the word embedding and softmax weights')
    parser.add_argument('--seed', type=int, default=1111,
                        help='random seed')
    parser.add_argument('--cuda', action='store_true',
                        help='use CUDA')
    parser.add_argument('--log-interval', type=int, default=1000000, metavar='N',
                        help='report interval')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--words', type=int, default='50',
                        help='number of words to generate')
    parser.add_argument('--mode', choices=['all', 'test'], default='all',
                        help='train and test / test only')
    parser.add_argument('--model_dir', type=str, default='RNNLM10_23_1240_N5',
                        help='directory name which has best model(at test only  mode)')
    parser.add_argument('--model_name', type=str, default='model_95.pth',
                        help='best model name(at test only  mode)')
    parser.add_argument('--ngrams', type=int, default=5,
                        help='select N for N-grams')

    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()

    torch.manual_seed(args.seed)

    vocab_path=file_path+'enwiki_vocab30000.txt'
    vocab = readVocab(vocab_path)
    ntokens = vocab.n_words

    #学習時
    if args.mode == 'all':
        train_file=file_path+'text8.txt'
        #train_file=file_path+'text8_mini.txt'

        #文字列→ID列に
        all_data=data_tokenize(train_file, vocab)

        #ID列からX, Yの組を作成して，学習データと検証データ作成
        train_data, val_data=make_data(all_data, args.ngrams)

        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

        criterion = nn.CrossEntropyLoss()

        lr = args.lr
        best_val_loss = None
        best_epoch = -1
        plot_train_loss=[]
        plot_val_loss=[]
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            for epoch in range(1, args.epochs+1):
                epoch_start_time = time.time()
                train_loss = train(ntokens, train_data)
                val_loss = evaluate(ntokens, val_data)
                plot_train_loss.append(train_loss)
                plot_val_loss.append(val_loss)

                print('-' * 89)
                print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                        'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                                   val_loss, math.exp(val_loss)))
                print('-' * 89)
                # Save the model if the validation loss is the best we've seen so far.
                if not best_val_loss or val_loss < best_val_loss:
                    best_epoch=epoch
                    best_weight=copy.deepcopy(model.state_dict())
                    best_val_loss = val_loss
                else:
                    # Anneal the learning rate if no improvement has been seen in the validation dataset.
                    lr /= 4.0
        except KeyboardInterrupt:
            print('-' * 89)
            if best_epoch >=0:
                print('Exiting from training early')
            else :
                exit()

        # Load the best saved model.
        model.load_state_dict(best_weight)

        #モデルとか結果とかを格納するディレクトリの作成
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)

        save_path=save_path+'_N'+str(args.ngrams) + '/'
        torch.save(model.state_dict(), save_path+'model_'+str(best_epoch)+'.pth')

        showPlot2(plot_train_loss, plot_val_loss)

    #すでにあるモデルをロードしてテスト
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

        save_path = file_path + args.model_dir +'/'

        model.load_state_dict(torch.load(save_path+args.model_name))

    #テストデータに対する予測と精度の計算
    model.eval()
    test_cloze=git_data_path+'center_cloze.txt'
    test_ans=git_data_path+'center_ans.txt'
    test_choi=git_data_path+'center_choices.txt'

    print("Reading Testdata...")
    test_data=readData(test_cloze, test_ans)
    choices=get_choices(test_choi)

    #前から予測スコア（方法A）
    #空所内1単語のみ（選択肢ありなし両方）
    print('\npreds by prob from top')
    '''
    #こっちは学習にパディングの処理とか入れてから？
    #いったん後回しにする
    前から予測用（1語限定）
    モデルの入力部分作成
        |___空所の直前まで（選択肢は見なくてOK）
        |___空所の前がnに満たない場合はどうする？
    モデルの出力する確率確認
        |___確率一覧を見て，最大の語（選択肢なし），選択肢のうち最大の語（選択肢あり）
    '''
    #print('one_word(use choices / not use choices)')
    #data=make_data_from_all_words(test_data, choices, all_words)
    #calc_acc(lang, data, model)

    #文スコア（方法B）
    #空所内1単語以上（選択肢あり）
    #空所内1単語のみ（選択肢ありなし両方）
    print('\npreds by sent score')
    print('Use choices')
    all_words=vocab.idx2word.values()
    data=make_data_for_sent_score(test_data, choices, one_word=True)
    calc_acc(vocab, data, model, args.ngrams)

    print('\nNot use choices, from all words')
    data=make_data_for_sent_score(test_data, choices, all_words)
    calc_acc(vocab, data, model, args.ngrams)
