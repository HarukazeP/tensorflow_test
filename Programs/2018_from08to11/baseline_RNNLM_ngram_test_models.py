# -*- coding: utf-8 -*-

'''
baseline_RNNLM_ngram_pretrain_vec.py から変更
ベストモデルを複数まとめてテスト用

動かしていたバージョン
python  : 3.5.2
pytorch : 2.0.4
gensim  : 3.5.0

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

import gensim
import numpy as np

#----- グローバル変数一覧 -----

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
git_data_path='../../Data/'
today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + 'RNNLM' + today_str

PAD_token = 0
UNK_token = 1

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
        self.word2idx = {"<PAD>":PAD_token, "<UNK>": UNK_token}
        self.idx2word = {PAD_token: "<PAD>", UNK_token: "<UNK>"}
        self.n_words = 2  # PAD と UNK

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


#Googleのword2vec読み取り
def get_weight_matrix(lang):
    print('Loading word vector ...')
    #ここのgensimの書き方がバージョンによって異なる
    vec_model = gensim.models.KeyedVectors.load_word2vec_format(file_path+'GoogleNews-vectors-negative300.bin', binary=True)
    # https://code.google.com/archive/p/word2vec/ ここからダウンロード&解凍

    weights_matrix = np.zeros((lang.n_words, args.emsize))

    for i, word in lang.idx2word.items():
        try:
            weights_matrix[i] = vec_model.wv[word]
        except KeyError:
            weights_matrix[i] = np.random.normal(size=(args.emsize, ))

    del vec_model
    #これメモリ解放的なことらしい、なくてもいいかも

    #パディングのところを初期化
    #Emneddingで引数のpad_index指定は、そこだけ更新(微分)しないらしい？
    weights_matrix[PAD_token]=np.zeros(args.emsize)

    return weights_matrix


#model.py内
class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""
    '''
    ntoken : 語彙数
    ninp   : embedingの次元数
    nhid   : 隠れ層の次元数
    nlayers: LSTMの層の数

    '''

    def __init__(self, ntoken, ninp, nhid, nlayers, weights_matrix, dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp, padding_idx=PAD_token)
        self.encoder.weight.data.copy_(torch.from_numpy(weights_matrix))
        self.rnn = nn.LSTM(ninp, nhid, nlayers, dropout=dropout, bidirectional=True)

        self.decoder = nn.Linear(nhid*N*2, ntoken) #(入力次元数, 出力次元数)

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

        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        #self.encoder.weight.data.uniform_(-initrange, initrange)
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
        return (weight.new_zeros(self.nlayers*2, bsz, self.nhid),
                weight.new_zeros(self.nlayers*2, bsz, self.nhid))






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
def input_ngram(cloze_sent, N):
    before=re.sub(r'{.*', '', cloze_sent)
    before=before.strip()
    if before=='':
        before="<PAD>"
    words=before.split(' ')
    length=len(words)
    if length < N:
        words=["<PAD>"]*(N-length)+words
    elif length > N:
        words=words[length-N:]

    return words


def get_ans_word(ans_sent):
    word=re.sub(r'.*{ ', '', ans_sent)
    word=re.sub(r' }.*', '', word)
    word=word.strip()

    return word


#ファイルから選択肢補充済みn-gramと答えのセット
def make_data_for_fw_score(data_pair, choices_lists, N):
    data=[]
    for sent, choices in zip(data_pair, choices_lists):
        flag=1
        for choice in choices:
            if(len(choice.split(' '))>1):
                flag=-1
                #選択肢に2語以上のものがあるときはflagが負
        if(flag>0):
            test_data=[]
            test_data.append(input_ngram(sent[0], N)) #ngram
            test_data.append(choices) #選択肢
            test_data.append(get_ans_word(sent[1])) #答え
            data.append(test_data)

    return data # [input_ngram, choices_list, ans_word]


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
        if(one_word):
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
def calc_acc_for_fw_score(lang, data_fw, model, N):
    line=0
    OK=0
    for one_data in data_fw:
        line+=1
        if line%50==0:
            print('line:',line)
        input_ngram=one_data[0]
        choices=one_data[1]
        ans_word=one_data[2]
        pred_word=get_best_word(lang, input_ngram, choices, model, N)
        if pred_word == ans_word:
            OK+=1
    print_score(line, OK)


#正答率の算出
def calc_acc_for_sent_score(lang, data, model, N):
    line=0
    OK=0
    for one_data in data:
        line+=1
        if line%50==0:
            print('line:',line)
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
def sent_to_ngram_pair(sent, N):
    pair=[]
    words=sent.split(' ')
    length=len(words)
    if length < N+1:
        words=["<PAD>"]*(N+1-length)+words
        length=N+1

    for i in range(length-N-1):
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


#ngramのペアからモデルの返す尤度をもとにスコアを算出
#TODO これ間違ってるのか，エラーになる
#RuntimeError: Expected hidden[0] size (2, 5, 200), got (2, 1, 200)
def calc_sent_score2(lang, ngram_pair, model):
    score=0
    batch=1
    #ほんとはbatch=1のはずだが，ngramと同じにしないとエラーでる
    hidden = model.init_hidden(batch)
    with torch.no_grad():
        for one_pair in ngram_pair:
            ids=sent_to_idxs(one_pair[0], lang)
            #zeros=[[0]*(N)]*(batch-1)
            #input_idx=[ids]+zeros
            input = torch.tensor(ids, dtype=torch.long).to(device)
            input = input.unsqueeze(0)  #(1, N)
            output, hidden_out = model(input, hidden)    #(5, 語彙数)
            probs=F.log_softmax(output.squeeze())
            word_idx=lang.check_word2idx(one_pair[1])
            score+=probs[word_idx].item()

    #返り値のスコアは文長で正規化する
    return score/len(ngram_pair)

#ngramのペアからモデルの返す尤度をもとにスコアを算出
def calc_sent_score(lang, ngram_pair, model):
    score=0
    batch=N
    #ほんとはbatch=1のはずだが，ngramと同じにしないとエラーでる
    hidden = model.init_hidden(batch)
    with torch.no_grad():
        for one_pair in ngram_pair:
            ids=sent_to_idxs(one_pair[0], lang)
            zeros=[[0]*(N)]*(batch-1)
            input_idx=[ids]+zeros
            input = torch.tensor(input_idx, dtype=torch.long).to(device)
            #input = input.unsqueeze(0)  #(1, N)
            output, hidden_out = model(input, hidden)    #(5, 語彙数)
            probs=F.log_softmax(output.squeeze(),dim=1)
            word_idx=lang.check_word2idx(one_pair[1])
            score+=probs[0][word_idx].item()

    #返り値のスコアは文長で正規化する
    if len(ngram_pair)==0:
        return -10000000.0
    return score/len(ngram_pair)


#1つの問題に対する，選択肢補充済み文複数から
#ベスト1文を返す
def get_best_sent(lang, sents, model, N):
    scores=[]
    sent_num=0
    best_sent=''
    for sent in sents:
        sent_num+=1
        if sent_num%5000==0:
            print('sent:',sent_num)
        ngram_pair=sent_to_ngram_pair(sent, N)
        #scoreは対数尤度 -inf ～ 0
        score=calc_sent_score(lang, ngram_pair, model)
        scores.append(score)

    return sents[scores.index(max(scores))]


def compare_choices(lang, probs, choices):
    scores=[]
    for word in choices:
        word_idx=lang.check_word2idx(word)
        scores.appened(probs[word_idx].item())

    return choices[scores.index(max(scores))]



#1つの問題に対する，選択肢補充済み文複数から
#ベスト1文を返す
def get_best_word(lang, ngram, choices, model, N):
    batch=N
    #ほんとはbatch=1のはずだが，ngramと同じにしないとエラーでる
    hidden = model.init_hidden(batch)
    with torch.no_grad():
        #print(ngram)
        ids=sent_to_idxs(ngram, lang)
        #print(ids)
        zeros=[[0]*(N)]*(batch-1)
        input_idx=[ids]+zeros
        print(input_idx)
        input = torch.tensor(input_idx, dtype=torch.long).to(device)
        #input = input.unsqueeze(0)  #(1, N)
        output, hidden_out = model(input, hidden)    #(5, 語彙数)
        probs=F.log_softmax(output.squeeze(),dim=1)

        best_word=compare_choices(lang, probs[0], choices)

    return best_word




#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--emsize', type=int, default=300,
                        help='size of word embeddings')
    parser.add_argument('--nhid', type=int, default=128,
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
    parser.add_argument('--mode', choices=['all','mini', 'test'], default='all',
                        help='train and test / test only')
    parser.add_argument('--model_dir', type=str, default='RNNLM10_23_1240_N5',
                        help='directory name which has best model(at test only  mode)')
    parser.add_argument('--model_name', type=str, default='model_95.pth',
                        help='best model name(at test only  mode)')
    #parser.add_argument('--ngrams', type=int, default=5, help='select N for N-grams')

    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()

    torch.manual_seed(args.seed)

    vocab_path=file_path+'enwiki_vocab30000_wordonly.txt'
    vocab = readVocab(vocab_path)
    ntokens = vocab.n_words
    all_words=vocab.idx2word.values()

    center_cloze=git_data_path+'center_cloze.txt'
    center_ans=git_data_path+'center_ans.txt'
    center_choi=git_data_path+'center_choices.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_ans=git_data_path+'microsoft_ans.txt'
    MS_choi=git_data_path+'microsoft_choices.txt'

    #print("Reading Testdata...")
    center_data=readData(center_cloze, center_ans)
    center_choices=get_choices(center_choi)

    MS_data=readData(MS_cloze, MS_ans)
    MS_choices=get_choices(MS_choi)

    #テスト時なのでembedding初期値読み込まなくていい
    weights_matrix=np.zeros((ntokens, args.emsize))

    dir0='RNNLM11_10_2200_biLSTM_N3'
    model0='model_7.pth'

    dir1='RNNLM11_07_2314_biLSTM_N4'
    model1='model_8.pth'

    dir2='RNNLM11_07_2304_biLSTM_N5'
    model2='model_7.pth'

    dir3='RNNLM11_11_1919_biLSTM_N7'
    model3='model_5.pth'




    files=[(dir0, model0), (dir1, model1), (dir2, model2), (dir3, model3)]

    for best_model in files:
        N=int(best_model[0][-1])

        model = RNNModel(ntokens, args.emsize, args.nhid, args.nlayers, weights_matrix, args.dropout, args.tied).to(device)

        save_path = file_path + best_model[0] +'/'

        model.load_state_dict(torch.load(save_path+best_model[1]))

        #テストデータに対する予測と精度の計算
        model.eval()

        #前から予測スコア（方法A）
        #空所内1単語のみ（選択肢ありなし両方）
        print('\npreds by forward score')
        print('Use choices(one_words)')
        print('center')
        data_fw=make_data_for_fw_score(center_data, center_choices, N)
        # data_fwは [input_ngram_list, choices_list, ans_word(str)] のリスト
        calc_acc_for_fw_score(vocab, data_fw, model, N)

        print('MS')
        data_fw=make_data_for_fw_score(MS_data, MS_choices, N)
        # data_fwは [input_ngram_list, choices_list, ans_word(str)] のリスト
        calc_acc_for_fw_score(vocab, data_fw, model, N)

        #文スコア（方法B）
        #空所内1単語以上（選択肢あり）
        #空所内1単語のみ（選択肢ありなし両方）
        print('\npreds by sent score')

        print('Use choices(one_words)')
        print('center')
        data=make_data_for_sent_score(center_data, center_choices, one_word=True)
        calc_acc_for_sent_score(vocab, data, model, N)

        print('MS')
        data=make_data_for_sent_score(MS_data, MS_choices, one_word=True)
        calc_acc_for_sent_score(vocab, data, model, N)


        print('Use choices(over one_words)')
        print('center')
        data=make_data_for_sent_score(center_data, center_choices, one_word=False)
        calc_acc_for_sent_score(vocab, data, model, N)

        print('MS')
        data=make_data_for_sent_score(MS_data, MS_choices, one_word=False)
        calc_acc_for_sent_score(vocab, data, model, N)

        '''
        print('\nNot use choices, from all words(one_words)')
        data=make_data_for_sent_score_from_all_words(center_data, center_choices, all_words)
        calc_acc_for_sent_score(vocab, data, model, N)
        '''
