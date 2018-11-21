# -*- coding: utf-8 -*-

'''
kerasを用いたseq2seqモデル
急遽実装
kerasチュートリアル→先生が編集→自分でテスト部分追加


Sequence to sequence example in Keras (character-level).

This script demonstrates how to implement a basic character-level
sequence-to-sequence model. We apply it to translating
short English sentences into short French sentences,
character-by-character. Note that it is fairly unusual to
do character-level machine translation, as word-level
models are more common in this domain.

# Summary of the algorithm

- We start with input sequences from a domain (e.g. English sentences)
    and corresponding target sequences from another domain
    (e.g. French sentences).
- An encoder LSTM turns input sequences to 2 state vectors
    (we keep the last LSTM state and discard the outputs).
- A decoder LSTM is trained to turn the target sequences into
    the same sequence but offset by one timestep in the future,
    a training process called "teacher forcing" in this context.
    Is uses as initial state the state vectors from the encoder.
    Effectively, the decoder learns to generate `targets[t+1...]`
    given `targets[...t]`, conditioned on the input sequence.
- In inference mode, when we want to decode unknown input sequences, we:
    - Encode the input sequence into state vectors
    - Start with a target sequence of size 1
        (just the start-of-sequence character)
    - Feed the state vectors and 1-char target sequence
        to the decoder to produce predictions for the next character
    - Sample the next character using these predictions
        (we simply use argmax).
    - Append the sampled character to the target sequence
    - Repeat until we generate the end-of-sequence character or we
        hit the character limit.

# Data download

English to French sentence pairs.
http://www.manythings.org/anki/fra-eng.zip

Lots of neat sentence pairs datasets can be found at:
http://www.manythings.org/anki/

# References

- Sequence to Sequence Learning with Neural Networks
    https://arxiv.org/abs/1409.3215
- Learning Phrase Representations using
    RNN Encoder-Decoder for Statistical Machine Translation
    https://arxiv.org/abs/1406.1078
'''
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, LSTM, Dense
from keras.models import load_model
import numpy as np

import sys
import random
import copy
import argparse
import re

# 自分で追加した変数
MAX_LENGTH = 200

#自分で追加
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test'], default='all')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--num_sample', type=int, default=160872)
    parser.add_argument('--max_L', type=int, default=200)
    parser.add_argument('--use_max_L', type=int, default=1)
    #TODO ほかにも引数必要に応じて追加
    return parser.parse_args()

args = get_args()

batch_size = 64  # Batch size for training.

#epochs = 100  # Number of epochs to train for.
#epochs = 3  # Number of epochs to train for.
epochs=args.epoch

latent_dim = 256  # Latent dimensionality of the encoding space.
#num_samples = 10000  # Number of samples to train on.
num_samples =  args.num_sample
#num_samples = 160872 #全行
# Path to the data txt file on disk.
data_path = '/home/ohtalab/niitsuma/keras/eng2fra/fra-eng/fra.txt'

if args.mode == 'mini':
    epochs = 3
    num_samples =  5000



# Vectorize the data.
input_texts = []
target_texts = []
input_characters = set()
input_characters.add('\t')
input_characters.add('\n')
# input_characters.add('\v')
# input_characters.add('\f')

target_characters = set()
target_characters.add('\t')
target_characters.add('\n')

# target_characters.add('\v')
# target_characters.add('\f')

def text_word_replace_v(text,s_rand):
    w_list=text.split()
    n_w=len(w_list)
    if n_w >= s_rand :
        #w_list[random.randint(0,n_w-1)]='\v'
        w_list[random.randint(0,n_w-1)]='\t'
        return ' '.join(w_list)
    else:
        return text

with open(data_path, 'r', encoding='utf-8') as f:
    lines = f.read().split('\n')
for line in lines[: min(num_samples, len(lines) - 1)]:
    input_text, target_text = line.split('\t')
    # We use "tab" as the "start sequence" character
    # for the targets, and "\n" as "end sequence" character.

    input_list=input_text.split()
    n_w=len(input_list)
    s_rand=5
    if n_w >= s_rand :
        #n_rand=int(int(n_w) / int(s_rand))
        n_rand=s_rand
    else:
        n_rand=1
    target_text = copy.copy(input_text)
    for k in range(n_rand):
        input_text  = text_word_replace_v(input_text,s_rand)

        # if n_w >= s_rand:
        #     print(input_text)
        #     sys.exit()

        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)

        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

input_characters = sorted(list(input_characters))
target_characters = sorted(list(target_characters))
num_encoder_tokens = len(input_characters)
num_decoder_tokens = len(target_characters)


max_encoder_seq_length = max([len(txt) for txt in input_texts])
max_decoder_seq_length = max([len(txt) for txt in target_texts])

if args.use_max_L==1 and not args.mode=='all':
    max_encoder_seq_length = max(args.max_L, max_encoder_seq_length)
    max_decoder_seq_length = max(args.max_L, max_decoder_seq_length)


print('Number of samples:', len(input_texts))
print('Number of unique input tokens:', num_encoder_tokens)
print('Number of unique output tokens:', num_decoder_tokens)
print('Max sequence length for inputs:', max_encoder_seq_length)
print('Max sequence length for outputs:', max_decoder_seq_length)

input_token_index = dict(
    [(char, i) for i, char in enumerate(input_characters)])
target_token_index = dict(
    [(char, i) for i, char in enumerate(target_characters)])

encoder_input_data = np.zeros(
    (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
    dtype='float32')
decoder_input_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')
decoder_target_data = np.zeros(
    (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
    dtype='float32')

for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
    for t, char in enumerate(input_text):
        encoder_input_data[i, t, input_token_index[char]] = 1.
    for t, char in enumerate(target_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t, target_token_index[char]] = 1.
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, target_token_index[char]] = 1.

# Define an input sequence and process it.
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# We discard `encoder_outputs` and only keep the states.
encoder_states = [state_h, state_c]

# Set up the decoder, using `encoder_states` as initial state.
decoder_inputs = Input(shape=(None, num_decoder_tokens))
# We set up our decoder to return full output sequences,
# and to return internal states as well. We don't use the
# return states in the training model, but we will use them in inference.
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                     initial_state=encoder_states)
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

# Define the model that will turn
# `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)



# Run training
if args.mode == 'all' or args.mode == 'mini':
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.2)
    # Save model
    model.save('/home/ohtalab/tamaki/M2/s2s_keras_max'+str(max_encoder_seq_length)+'_ep'+str(epochs)+'.h5')

else :
    print('load model')
    model=load_model('/home/ohtalab/tamaki/M2/s2s.h5')
# json_string = model.to_json()
# print(json_string)
# import json
# with open('s2s.json', 'w') as f:
#     json.dump(json_string, f)
# #sys.exit()


# Next: inference mode (sampling).
# Here's the drill:
# 1) encode input and retrieve initial decoder state
# 2) run one step of decoder with this initial state
# and a "start of sequence" token as target.
# Output will be the next target token
# 3) Repeat with the current target token and current states

# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(
    decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model(
    [decoder_inputs] + decoder_states_inputs,
    [decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict(
    (i, char) for char, i in target_token_index.items())


def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq] + states_value)

        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or
           len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.

        # Update states
        states_value = [h, c]

    return decoded_sentence



#ペアじゃなくて単独で読み取るやつ
def readData2(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())

    return data



def get_choices(file_name):
    print("Reading data...")
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=line.strip()
            line=get_cloze(line)
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices

def get_cloze(line):
    line=re.sub(r'.*{ ', '', line)
    line=re.sub(r' }.*', '', line)

    return line


def remove_cloze_mark(line):
    line=line.replace('{', '')
    line=line.replace('}', '')
    line = re.sub(r'[ ]+', ' ', line)

    return line.strip()

def make_sents_without_cloze_mark(sentence, choices):
    sents=[]
    before=re.sub(r'{.*', '', sentence)
    after=re.sub(r'.*}', '', sentence)
    for choice in choices:
        tmp=before + choice + after
        tmp = re.sub(r'[ ]+', ' ', tmp)
        sents.append(tmp.strip())

    return sents


def make_choices(cloze_path, choices_path):
    choices_sents_list=[]
    cloze_sents=readData2(cloze_path)
    choices=get_choices(choices_path)
    for sent, choi in zip(cloze_sents, choices):
        sents=make_sents_without_cloze_mark(sent, choi)
        choices_sents_list.append(sents)

    return choices_sents_list


def get_best_sent(input_seq, choices_sents):
    '''
    input_seq       : id列   ... 空所を\tにしてある文1つ
    choices_sents   : 文字列 ... 選択肢を補充した文4つor5つ
    '''
    scores=[]
    for c_sent in choices_sents:
        score=0
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        for i in range(len(c_sent)):
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            try:
                char_id=target_token_index[c_sent[i]]
            except KeyError:
                char_id=0 #TODO これで大丈夫？

            sampled_token_index=char_id
            score += output_tokens[0, -1, :][char_id]

            sampled_char = reverse_target_char_index[sampled_token_index]

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        sent_score=score/len(c_sent)
        scores.append(sent_score)

    return choices_sents[scores.index(max(scores))]

def read_input_test_data(data_path):
    test_input_texts = []
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines:
        line=line.replace('{ }', '\t')
        line = re.sub(r'[ ]+', ' ', line)
        test_input_text = line
        test_input_texts.append(test_input_text)

    encoder_test_input_data = np.zeros(
        (len(test_input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')

    for i ,test_input_text in enumerate(test_input_texts):
        for t, char in enumerate(test_input_text):
            try:
                char_id=input_token_index[char]
            except KeyError:
                char_id=0 #TODO これで大丈夫？

            encoder_test_input_data[i, t, char_id] = 1.

    return len(test_input_texts), encoder_test_input_data

def calc_acc(preds_sentences, ans_sentences):
    line_num=0
    allOK=0

    for pred, ans in zip(preds_sentences, ans_sentences):
        ans=remove_cloze_mark(ans)
        line_num+=1
        if pred == ans:
            allOK+=1

    print('  acc(all): ', '{0:.2f}'.format(1.0*allOK/line_num*100),' %')
    print('  all: ', allOK)
    print(' line: ',line_num)


for seq_index in range(0,10):
    # Take one sequence (part of the training set)
    # for trying out decoding.

    i_seq=random.randint(300,len(input_texts)-1)

    #input_seq = encoder_input_data[seq_index: seq_index + 1]
    input_seq = encoder_input_data[i_seq: i_seq + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-')
    #print('Input sentence:', input_texts[seq_index])
    print('Input sentence:', input_texts[i_seq])
    print('Decoded sentence:', decoded_sentence)



git_data_path='../../Data/'
center_cloze=git_data_path+'center_cloze.txt'
center_ans=git_data_path+'center_ans.txt'
center_choi=git_data_path+'center_choices.txt'

MS_cloze=git_data_path+'microsoft_cloze.txt'
MS_ans=git_data_path+'microsoft_ans.txt'
MS_choi=git_data_path+'microsoft_choices.txt'


#center
text_num, test_input=read_input_test_data(center_cloze)
ans_sents=readData2(center_ans)
preds_sents=[]
choices_sents_list=make_choices(center_cloze, center_choi)
print('Test center')
for i in range(text_num-1):
    input_seq = test_input[i: i + 1]
    preds_sent=get_best_sent(input_seq, choices_sents_list[i])
    preds_sents.append(preds_sent)
calc_acc(preds_sents, ans_sents)

#MS
text_num, test_input=read_input_test_data(MS_cloze)
ans_sents=readData2(MS_ans)
preds_sents=[]
choices_sents_list=make_choices(MS_cloze, MS_choi)
print('Test MS')
for i in range(text_num-1):
    input_seq = test_input[i: i + 1]
    preds_sent=get_best_sent(input_seq, choices_sents_list[i])
    preds_sents.append(preds_sent)
calc_acc(preds_sents, ans_sents)
