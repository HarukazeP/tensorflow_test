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

import time
import math

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker

import os
import argparse
import copy

#----- グローバル変数一覧 -----

#自分で定義したグローバル関数とか
file_path='../../../pytorch_data/'
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

def data_tokenize(file, lang):
    length=0
    all_ids=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            line=normalizeString(line)
            words = line.split() + ['<eos>']
            for word in words:
                all_ids.append(lang.check_word2idx(word))
                length += 1

    border=int(length*9/10)

    train_tokens=torch.tensor(all_ids[:border], dtype=torch.long, device=device)
    val_tokens=torch.tensor(all_ids[border:], dtype=torch.long, device=device)

    return train_tokens, val_tokens

#model.py内
#TODO いろいろ変更
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
        self.decoder = nn.Linear(nhid, ntoken)

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
        output = self.drop(output)
        decoded = self.decoder(output.view(output.size(0)*output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)

# Starting from sequential data, batchify arranges the dataset into columns.
# For instance, with the alphabet as the sequence and batch size 4, we'd get
# ┌ a g m s ┐
# │ b h n t │
# │ c i o u │
# │ d j p v │
# │ e k q w │
# └ f l r x ┘.
# These columns are treated as independent by the model, which means that the
# dependence of e. g. 'g' on 'f' can not be learned, but allows more efficient
# batch processing.
#TODO いろいろ変更
def batchify(data, bsz):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)  #バッチの余りになる部分を切り捨て
    data = data.view(nbatch, -1)    #bsz単語ごとに分割？

    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)



###############################################################################
# Training code
###############################################################################

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


# get_batch subdivides the source data into chunks of length args.bptt.
# If source is equal to the example output of the batchify function, with
# a bptt-limit of 2, we'd get the following two Variables for i = 0:
# ┌ a g m s ┐ ┌ b h n t ┐
# └ b h n t ┘ └ c i o u ┘
# Note that despite the name of the function, the subdivison of data is not
# done along the batch dimension (i.e. dimension 1), since that was handled
# by the batchify function. The chunks are along dimension 0, corresponding
# to the seq_len dimension in the LSTM.
def get_batch(source, i):
    seq_len = min(args.bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target


def evaluate(ntokens, data_source):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0.
    hidden = model.init_hidden(eval_batch_size)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, args.bptt):
            data, targets = get_batch(data_source, i)
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
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):
        data, targets = get_batch(train_data, i)
        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
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
                epoch, batch, len(train_data) // args.bptt, lr,
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
    parser.add_argument('--epochs', type=int, default=40,
                        help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=20, metavar='N',
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
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='report interval')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='temperature - higher will increase diversity')
    parser.add_argument('--mode', choices=['all', 'test'], default='all',
                        help='train and test / test only')
    parser.add_argument('--model_dir', type=str, default='RNNLM10_17_1745',
                        help='directory name which has best model(at test only  mode)')
    parser.add_argument('--model_name', type=str, default='model_95.pth',
                        help='best model name(at test only  mode)')
    parser.add_argument('--ngrams', type=int, default=1,
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
        #train_file=file_path+'text8.txt'
        train_file=file_path+'text8_mini.txt'

        train_tk, val_tk=data_tokenize(train_file, vocab)

        eval_batch_size = 10
        train_data = batchify(train_tk, args.batch_size)
        val_data = batchify(val_tk, eval_batch_size)

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

        save_path=save_path+'/'
        torch.save(model.state_dict(), save_path+'model_'+str(best_epoch)+'.pth')

        showPlot2(plot_train_loss, plot_val_loss)

    #すでにあるモデルをロードしてテスト
    else:
        model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied).to(device)

        save_path = file_path + args.model_dir +'/'

        model.load_state_dict(torch.load(save_path+args.model_name))

    #テスト時
    model.eval()

    choi_file=file_path+'aaaaaa.txt'
    ans_file=file_path+'aaaaaa.txt'

    #TODO これ試しにテストしてるだけ
    n_gram=args.ngrams
    hidden = model.init_hidden(n_gram)
    input = torch.randint(ntokens, (n_gram, 1), dtype=torch.long).to(device)
    with torch.no_grad():  # no tracking history
        for i in range(args.words):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            word = corpus.dictionary.idx2word[word_idx]

            if i % args.log_interval == 0:
                print('| Generated {}/{} words'.format(i, args.words))
    print(output.size())
    print(word_weights.size())
    print(torch.multinomial(word_weights, 1).size())
