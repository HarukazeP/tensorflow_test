# -*- coding: utf-8 -*-

'''

M1のときにやってた双方向ベクトル予測モデル
CLOTHデータセットで学習

CLOTH用の前処理

当時はやってなかった補充文スコアでのテストもやる
テストデータも変わってる

#TODO まだ未作成、まだ何も手つけてない
以下，以前のプログラムコピペしたやつ

#かなり書き直した方がよさそう？
実行効率的な意味で
少なくてもmodel.fit()とチェックポイントのやつは使いたい

-------------------------------------------

動かしていたバージョン
python  : 3.5.2

fasttextでベクトルにして学習するモデル
embeddingレイヤーでfasttextのベクトルを利用
gensimライブラリは使わない
単語のベクトル成分を予測する回帰モデル

python    : 2.7.12
keras     : 2.0.4
gensim    : 3.0.1
tensorflow: 1.1.0

プログラム全体の構成
    ・グローバル変数一覧
    ・関数群
    ・いわゆるmain部みたいなの

プログラム全体の流れ
    0.いろいろ前準備
    1.学習データの前処理
    2.fasttextのロードと辞書の作成
    3.モデルの定義
    4.モデルの学習
    5.val_loss最小モデルのロード
    6.テストの実行
    7.結果まとめの出力


memo
all start 2018/1/22/16:19
train end
2018-01-25 02:10:35.310248
Loading  ./2018_01_22_1619epoch100_e100_w5_mul_lstm_den1_2/Model_8/my_model.json
  → 2days, 10hour
sudo pip install pydot graphviz

sudo apt-get autoremove --purge linux-headers-4.4.0-21-generic

'''


'''
ここkerasの古いプログラムのやつ
    from __future__ import print_function
    from keras.models import Sequential, Model
    from keras.models import model_from_json
    from keras.layers import Dense, Activation, Input, Embedding
    from keras.layers import LSTM
    from keras.layers import add, concatenate, multiply
    from keras.optimizers import RMSprop
    from keras.utils.data_utils import get_file
    from keras.utils.vis_utils import plot_model
    import keras
    import numpy as np
    import re
    import sys
    import datetime
    import os
    import os.path
    import matplotlib
    matplotlib.use('Agg')    #これをpyplotより先に書くことでサーバでも動くようにしている
    import matplotlib.pyplot as plt
    import subprocess

    import unicodedata
    import string
    import nltk


    #----- グローバル変数一覧 -----
    my_epoch=100
    vec_size=100
    maxlen_words = 5
    KeyError_set=set()
    today_str=''
    tmp_vec_dict=dict()

    #----- グローバル変数一覧 -----
    file_path='../../../pytorch_data/'
    git_data_path='../../Data/'
    CLOTH_path = file_path+'CLOTH_for_model/'

    today1=datetime.datetime.today()
    today_str=today1.strftime('%m_%d_%H%M')
    save_path=file_path + today_str
    PAD_token = 0
    UNK_token = 1
    NUM_token = 2

    #事前処理いろいろ
    print('Start: '+today_str)



    #----- 関数群 -----

    #時間表示
    def print_time(str1):
        today=datetime.datetime.today()
        print(str1)
        print(today)
        return today


    #半角カナとか特殊記号とかを正規化
    # Ａ→A，Ⅲ→III，①→1とかそういうの
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
        )


    #学習データやテストデータへの前処理
    def preprocess_line(before_line):
        after_line = re.sub(r'[ ]+', ' ', after_line)

        return after_line

    def hoge():
            sent_tokens=[]
            s = unicodeToAscii(s)
            s = re.sub(r'[ ]+', ' ', s)
            s = s.strip()
            s = re.sub(r'{.+}', CLZ_word, s)
            tokens=nltk.word_tokenize(s)
            symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
            num_tag=("LS", "CD")
            tagged = nltk.pos_tag(tokens)
            for word, tag in tagged:
                if tag in symbol_tag:
                    pass
                    #記号は無視
                elif tag in num_tag:
                    sent_tokens.append('NUM')
                else:
                    sent_tokens.append(word.lower())



    #listの各要素を単語で連結してstring型で返す
    def list_to_sent(list_line, start, end):
        sent=' '.join(list_line[start:end])
        return sent


    #学習データへの前処理を行う
    #小文字化，アルファベット以外の文字の削除，1万単語ごとに分割
    def preprocess(train_path):
        max_len=50000
        new_path=train_path[:-4]+'_preprpcessed'+str(max_len)+'.txt'
        if os.path.exists(new_path)==False:

            print('Preprpcessing training data...')
            text=''
            text_len=0
            i=0
            with open(train_path) as f_in:
                with open(new_path, 'w') as f_out:
                    for line in f_in:
                        #この前処理はtext8とかの前処理と同じ
                        line=preprocess_line(line)
                        line_list=line.split(' ')
                        line_len=len(line_list)
                        #max_len以下の時は連結して次へ
                        if(text_len+line_len <= max_len):
                            if(text_len==0):
                                text=line
                            else:
                                text=text+' '+line
                            text_len=text_len+line_len
                        #max_lenより長いときはmax_len単語ごとに区切ってファイルへ書き込み
                        else:
                            while (line_len>max_len):
                                if(text_len==0):
                                    text=list_to_sent(line_list,0,max_len)
                                else:
                                    text=text+' '+list_to_sent(line_list,0,max_len-text_len)
                                f_out.write(text+'\n')
                                text=''
                                text_len=0
                                #残りの更新
                                line_list=line_list[max_len-text_len+1:]
                                line_len=len(line_list)
                            #while 終わり（1行の末尾の処理）
                            #余りは次の行と連結
                            text=list_to_sent(line_list,0,line_len)
                            text_len=line_len
                    #for終わり（ファイルの最後の行の処理）
                    if text_len!=0:
                        text=preprocess_line(text)
                        f_out.write(text+'\n')
                    print('total '+str(i)+' line\n')
                    print_time('preprpcess end')

        return new_path



    #単語から辞書IDを返す
    def search_word_indices(word, word_to_id):
        if word in word_to_id:
            return word_to_id[word]
        else:
            return word_to_id['#OTHER']


    #1行の文字列を学習データの形式に変換
    def make_train_data(line, len_words, word_to_id, vec_dict, ft_path, bin_path):
        #TODO ここで前処理？
        line=line.strip()
        text_list=line.split(' ')
        f_sentences = list()
        r_sentences = list()
        next_words = list()
        step=3
        len_text=len(text_list)
        if (len_text - maxlen_words*2 -1) > 0:
            for i in range(0, len_text - maxlen_words*2 -1, step):
                f=text_list[i: i + maxlen_words]
                r=text_list[i + maxlen_words+1: i + maxlen_words+1+maxlen_words]
                n=text_list[i + maxlen_words]
                f_sentences.append(f)
                r_sentences.append(r[::-1]) #逆順のリスト
                next_words.append(n)
            len_sent=len(f_sentences)

            f_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
            r_X = np.zeros((len_sent, maxlen_words), dtype=np.int)
            Y = np.zeros((len_sent, vec_size), dtype=np.float32)
            for i, sentence in enumerate(f_sentences):
                Y[i] = get_ft_vec(next_words[i], vec_dict, ft_path, bin_path)
                for t, word in enumerate(sentence):
                    f_X[i, t] = search_word_indices(word, word_to_id)


            for i, sentence in enumerate(r_sentences):
                for t, word in enumerate(sentence):
                    r_X[i, t] = search_word_indices(word, word_to_id)

        return f_X, r_X, Y


    #loss, val_lossの追加更新
    def conect_hist(list_loss, list_val_loss, new_history):
        list_loss.extend(new_history.history['loss'])
        list_val_loss.extend(new_history.history['val_loss'])


    #1行10000単語までのファイルから1行ずつ1回学習する
    #lossやval_lossは各行の学習結果の中央値を返す
    def model_fit_once(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path):
        tmp_loss_list=list()
        tmp_val_loss_list=list()

        with open(train_path) as f:
            for line in f:
                line = re.sub(r'[ ]+', ' ', line)
                if line.count(' ')>maxlen_words*10:
                    f_trainX, r_trainX, trainY = make_train_data(line, len_words, word_to_id, vec_dict, ft_path, bin_path)
                    tmp_hist=my_model.fit([f_trainX,r_trainX], trainY, batch_size=128, epochs=1, validation_split=0.1)
                    conect_hist(tmp_loss_list, tmp_val_loss_list, tmp_hist)

        loss=np.median(np.array(tmp_loss_list, dtype=np.float32))
        val_loss=np.median(np.array(tmp_val_loss_list, dtype=np.float32))
        with open(today_str+'loss.txt', 'a') as f_loss:
                f_loss.write('loss='+str(loss)+'  , val_loss='+str(val_loss)+'\n')
        return loss, val_loss


    #my_epochの数だけ学習をくりかえす
    def model_fit_loop(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path):
        list_loss=list()
        list_val_loss=list()
        for ep_i in range(my_epoch):
            print('\nEPOCH='+str(ep_i+1)+'/'+str(my_epoch)+'\n')
            loss, val_loss=model_fit_once(train_path, my_model,len_words, word_to_id, vec_dict, ft_path, bin_path)
            list_loss.append(loss)
            list_val_loss.append(val_loss)

            #モデルの保存
            dir_name=today_str+'Model_'+str(ep_i+1)
            os.mkdir(dir_name)

            model_json_str = my_model.to_json()
            file_model=dir_name+'/my_model'
            open(file_model+'.json', 'w').write(model_json_str)
            my_model.save_weights(file_model+'.h5')


        return list_loss, list_val_loss


    # 損失の履歴をプロット
    def plot_loss(list_loss, list_val_loss, title='model loss'):
        plt.plot(list_loss, color='blue', marker='o', label='loss')
        plt.plot(list_val_loss, color='green', marker='o', label='val_loss')
        plt.title(title)
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(today_str+'loss_graph.png')
        #plt.show()は行うとプログラム中断されるからNG


    #テストデータの前準備
    def prepare_test(test_path, ch_path):

        th_len =maxlen_words/2    #テストの際の長さの閾値
        test_f_sentences = list()
        test_r_sentences = list()

        #テストデータへの読み込みと前処理
        #テストデータは学習データと異なり容量大きくないので一気に読み込んでいる
        #テストデータは1行1問で1行に<>が1つのみ
        test_file = open(test_path)
        test_data = test_file.read().lower().replace('\r','')
        test_file.close()

        ch_file= open(ch_path)
        ch_data= ch_file.read().lower().replace('\r','')
        ch_file.close()

        test_lines = test_data.split('\n')

        ch_lines = ch_data.split('\n')
        ans_list=list()
        ch_list=list()
        line_num=0

        for line in test_lines:
            if (line.count('<')*line.count('>')==1):
                mark_start=line.find('<')
                mark_end=line.find('>')
                ch_tmp_line=ch_lines[line_num]

                before=line[:mark_start]
                after=line[mark_end+1:]
                ans=line[mark_start+1:mark_end]
                choi=ch_tmp_line[ch_tmp_line.find('<')+1:ch_tmp_line.find('>')]

                before=preprocess_line(before)
                after=preprocess_line(after)
                ans=preprocess_line(ans)
                choices=choi.split(' ### ')
                flag=0
                tmp_choi=''
                for x in choices:
                    x=preprocess_line(x)
                    tmp_choi=tmp_choi+x+' ### '
                    if x.count(' ')>0:
                        flag=-1
                if(flag==0):
                    test_f_line=before.split(' ')
                    test_r_line=after.split(' ')
                    if (len(test_f_line)>=th_len) and (len(test_r_line)>=th_len):
                        if (len(test_f_line)>maxlen_words):
                            test_f_line=test_f_line[-1*maxlen_words:]
                        if (len(test_r_line)>maxlen_words):
                            test_r_line=test_r_line[:maxlen_words]
                        test_f_sentences.append(test_f_line)
                        test_r_sentences.append(test_r_line[::-1])
                        #テスト対象のデータの答えと選択肢をリストに格納
                        ans_list.append(ans)
                        choi=tmp_choi[:-5]  #末尾のシャープとかを削除
                        ch_list.append(choi)
                        #テスト対象となるデータのみを出力
                        with open(today_str+'testdata.txt', 'a') as data:
                            data.write(line+'\n')
            line_num+=1

        return test_f_sentences, test_r_sentences, ans_list, ch_list



    #テストデータのベクトル化
    def make_test_data(f_sent, r_sent, word_to_id):
        test_f_x = np.zeros((1, maxlen_words))
        test_r_x = np.zeros((1, maxlen_words))
        for t, word in enumerate(f_sent):
            tmp_index = search_word_indices(word, word_to_id)
            if(len(f_sent)<maxlen_words):
                test_f_x[0, t+maxlen_words-len(f_sent)] = tmp_index
            else:
                test_f_x[0, t] = tmp_index
        for t, word in enumerate(r_sent):
            tmp_index = search_word_indices(word, word_to_id)
            if(len(f_sent)<maxlen_words):
                test_r_x[0, t+maxlen_words-len(r_sent)] = tmp_index
            else:
                test_r_x[0, t] = tmp_index
        return test_f_x, test_r_x


    #2つのベクトルのコサイン類似度を返す
    def calc_similarity(pred_vec, ans_vec):
        len_p=np.linalg.norm(pred_vec)
        len_a=np.linalg.norm(ans_vec)
        if len_p==0 or len_a==0:
            return 0.0
        return np.dot(pred_vec/len_p, ans_vec/len_a)


    #全単語の中からベクトルの類似度の高い順にファイル出力（あとで考察用）し，
    #上位1語とその類似度，選択肢の各語の順位と類似度を返す
    def print_and_get_rank(pred_vec, choices, fname, vec_dict, ft_path, bin_path, id_to_word):
        dict_all=dict()
        dict_ch=dict()
        choi_list=choices.split(' ### ')
        for i in range(len_words):
            if i!=0:
                word=id_to_word[i]
                dict_all[word]=calc_similarity(pred_vec, get_ft_vec(word, vec_dict, ft_path, bin_path))
        for x in choi_list:
            dict_ch[x]=calc_similarity(pred_vec, get_ft_vec(x, vec_dict, ft_path, bin_path))
        list_ch = sorted(dict_ch.items(), key=lambda x: x[1], reverse=True)
        list_all = sorted(dict_all.items(), key=lambda x: x[1], reverse=True)
        with open(fname, 'a') as file:
            for w,sim in list_all:
                str=w+ ' ### '
                file.write(str)
            file.write('\n')
        return (list_all[0], list_ch)
        #返り値は(単語, 類似度), {(単語, 類似度),(単語, 類似度)...}


    #単語とランクリストから単語の順位をstring型で返す
    def word_to_rank(word, ra_list):
        str_num=''
        if word in ra_list:
            str_num=str(ra_list.index(word))
        else:
            #無いときは-1
            str_num='-1'

        return str_num


    #全単語の内類似度1語の正誤をファイル書き込み，正誤結果を返す
    def calc_rank1word(top, ans, ra_list, ans_sim):
        rank_top=word_to_rank(top[0], ra_list)
        rank_ans=word_to_rank(ans, ra_list)
        out=''
        with open(today_str+'rankOK.txt', 'a') as rOK:
            with open(today_str+'rankNG.txt', 'a') as rNG:
                out='pred= ('+top[0]+', '+rank_top+', '+str(top[1])+')   '+'ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')\n'
                if top[0]==ans:
                    rOK.write(out)
                    OK_num=1
                else:
                    rNG.write(out)
                    OK_num=0
        return OK_num


    #選択肢で選んだ際の正誤をファイル書き込み，正誤結果を返す
    #choices=[(単語,類似度),(単語,類似度), ...]の形式
    def calc_rank4choices(choices, ans, ra_list, ans_sim):
        pred=choices[0][0]
        rank_ans=word_to_rank(ans, ra_list)
        out='ans= ('+ans+', '+rank_ans+', '+str(ans_sim)+')    '+'choices='
        for word,sim in choices:
            out=out+'('+word+', '+word_to_rank(word, ra_list)+', '+str(sim)+')  '
        with open(today_str+'choicesOK.txt', 'a') as cOK:
            with open(today_str+'choicesNG.txt', 'a') as cNG:
                out=out+'\n'
                if pred==ans:
                    cOK.write(out)
                    OK_num=1
                else:
                    cNG.write(out)
                    OK_num=0
        return OK_num


    #正解率の計算結果を文字列で返す
    def calc_acc(rank_file, ans_list, preds_list, top_list, choice_list, vec_dict, ft_path, bin_path):
        sent_i=0
        rankOK=0
        choiOK=0

        with open(rank_file,'r') as rank:
            for line in rank:
                rank_line=line.lower().replace('\n','').replace('\r','')
                rank_list=rank_line.split(' ### ')
                ans=ans_list[sent_i]
                ans_sim=calc_similarity(preds_list[sent_i], get_ft_vec(ans, vec_dict, ft_path, bin_path))
                rankOK+=calc_rank1word(top_list[sent_i], ans, rank_list, ans_sim)
                choiOK+=calc_rank4choices(choice_list[sent_i], ans, rank_list, ans_sim)
                sent_i+=1

        rank_acc=1.0*rankOK/sent_i
        choi_acc=1.0*choiOK/sent_i

        rankNG=sent_i - rankOK
        choiNG=sent_i - choiOK

        rank_result='rank: '+str(rank_acc)+' ( OK: '+str(rankOK)+'   NG: '+str(rankNG)+' )\n'
        choi_result='choi: '+str(choi_acc)+' ( OK: '+str(choiOK)+'   NG: '+str(choiNG)+' )\n'

        result=rank_result+choi_result

        return result


    #テスト
    def model_test(model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word):
        #テストデータの前準備
        f_sent, r_sent, ans_list, ch_list = prepare_test(test_path, ch_path)
        sent_num=len(f_sent)
        preds_list=list()
        top_list=list()
        choice_list=list()
        #テストの実行
        for i in range(sent_num):
            f_testX, r_testX = make_test_data(f_sent[i], r_sent[i], word_to_id)
            preds = min_model.predict([f_testX, r_testX], verbose=0)
            preds_list.append(preds)
            #予測結果の格納
            rank_file=today_str+'rank.txt'
            tmp=print_and_get_rank(preds, ch_list[i], rank_file, vec_dict, ft_path, bin_path, id_to_word)
            top=tmp[0]
            choice=tmp[1]
            top_list.append(top)
            choice_list.append(choice)
        #正解率の計算，ファイル出力
        result_str=calc_acc(rank_file, ans_list, preds_list, top_list, choice_list, vec_dict, ft_path, bin_path)

        return result_str



    #----- main部 -----
    if __name__ == '__main__':

        #学習データの候補


        # 1.学習データの前処理など
        tmp_path = train_text8     #使用する学習データ
        print('Loading  '+tmp_path)
        #train_path=preprocess(tmp_path)
        train_path='../corpus/text8_preprpcessed50000_small.txt'


        end_data=print_time('prepare data and fasttext end')



        # 3.モデルの定義
        my_model=build_model(len_words, embedding_matrix)



        # 4.モデルの学習
        loss, val_loss=model_fit_loop(train_path, my_model, len_words, word_to_id, vec_dict, ft_path, bin_path)
        plot_loss(loss, val_loss)

        end_train=print_time('train end')



        # 5.val_loss最小モデルのロード
        min_i=np.array(val_loss).argmin()

        min_model_file=today_str+'Model_'+str(min_i+1)+'/my_model.json'
        min_weight_file=today_str+'Model_'+str(min_i+1)+'/my_model.h5'
        print('Loading  '+min_model_file)

        json_string = open(min_model_file).read()
        min_model = model_from_json(json_string)
        min_model.load_weights(min_weight_file)
        optimizer = RMSprop()
        min_model.compile(loss='mean_squared_error', optimizer=optimizer)

        #min_model.summary(print_fn=myprint)
        #summaryをファイル出力したいけどこれはうまくいかないようす
        plot_model(min_model, to_file=today_str+'model.png', show_shapes=True)

        end_load=print_time('Load min_model end')



        # 6.テストの実行
        test_path = '../corpus/ans_all2000_2016.txt'     #答えつきテストデータ
        ch_path= '../corpus/choi_all2000_2016.txt'     #選択肢つきテストデータ

        result=model_test(min_model, test_path, ch_path, word_to_id, vec_dict, ft_path, bin_path, id_to_word)
        print(result)

        with open(today_str+'keyerror_words.txt', 'w') as f_key:
            for word in KeyError_set:
                f_key.write(word+'\n')


        end_test=print_time('test end')



        #7.実行結果まとめのファイル書き込み
        #下記内容をファイルにまとめて出力
        '''
        ・実行したプログラム名
        ・実施日時（開始時刻）
        ・読み込んだ学習データ
        ・単語数
        ・全学習回数
        ・val_loss最小の学習回数

        ・テスト結果

        ・modelの概要

        ・学習データの前処理，辞書の作成ににかかった時間
        ・fasttextのロードとembedding_matrixの作成にかかった時間
        ・学習にかかった時間（ベクトル化も含む）
        ・val_loss最小モデルのロードにかかった時間
        ・テストにかかった時間（ベクトル化，正解率とかも含む）
        ・全合計かかった時間
        '''

        #実行結果のあれこれをファイル書き込み
        min_model.summary()

        with open(today_str+'summary.txt', 'a') as f:
            f.write('Result of '+program_name+'\n\n')

            f.write('start_time = '+ start_time_str+'\n')
            f.write('epoch = '+str(my_epoch)+'\n')
            f.write('train_data = '+ train_path+'\n')
            f.write('kind of words ='+str(len_words)+'\n')
            f.write('min_model = '+ min_model_file+'\n\n')

            f.write('result\n'+ result+'\n')

            f.write('TIME prepare data and fasttext= '+ str(end_data-start_time)+'\n')
            f.write('TIME train = '+ str(end_train-end_data)+'\n')
            f.write('TIME load min_model = '+ str(end_load-end_train)+'\n')
            f.write('TIME test = '+ str(end_test-end_load)+'\n\n')

            end_time=print_time('all end')
            f.write('TIME total = '+ str(end_time-start_time)+'\n')

'''
#-------------------------------------------------------
#ここからMPNetのやつ


'''
動かしていたバージョン
python  : 3.5.2 / 3.6.5
keras   : 0.2.0
gensim  : 3.1.0 / 3.5.0


'''


from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import string
import re
import random
import datetime
import time
import math
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import os
import argparse
import collections

import gensim
import nltk

from keras import regularizers
from keras import backend as K
from keras.models import Model, model_from_json, load_model
from keras.layers import Dense, Embedding, GRU, Bidirectional, Conv1D, GlobalMaxPooling1D
from keras.layers import Add, Multiply, Input, Activation, Reshape, Concatenate
from keras.layers.normalization import BatchNormalization
from keras.engine.topology import Layer
from keras.utils.vis_utils import plot_model
from keras.activations import softmax, sigmoid
from keras.callbacks import ModelCheckpoint
from keras import optimizers

from glob import glob
import kenlm

#----- グローバル変数一覧 -----
MAX_LENGTH = 80
C_MAXLEN = 6
HIDDEN_DIM = 128
EMB_DIM = 300
BATCH_SIZE = 128

file_path='../../../pytorch_data/'
git_data_path='../../Data/'
CLOTH_path = file_path+'CLOTH_for_model/'
KenLM_path='/media/tamaki/HDCL-UT/tamaki/M2/kenlm_models/all_words/'

today1=datetime.datetime.today()
today_str=today1.strftime('%m_%d_%H%M')
save_path=file_path + today_str
PAD_token = 0
UNK_token = 1
CLZ_token = 2
NUM_token = 3

CLZ_word = 'XXXX'

#事前処理いろいろ
print('Start: '+today_str)


#----- 関数群 -----

###########################
# 1.データの準備，データ変換
###########################

#語彙に関するクラス
class Lang:
    def __init__(self):
        self.word2index = {"<UNK>": UNK_token}
        self.index2word = {PAD_token: "PAD", UNK_token: "<UNK>", CLZ_token: "CLZ", NUM_token: "NUM"}
        self.n_words = 4  # PAD, UNK, CLZ, NUM

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


#与えた語彙読み込み
def readVocab(file):
    lang = Lang()
    print("Reading vocab...")
    with open(file, encoding='utf-8') as f:
        for line in f:
            lang.addSentence(line.strip())
    #print("Vocab: %s" % lang.n_words)

    return lang


#空所つき英文読み取り
def readCloze(file):
    #print("Reading data...")
    data=[]
    with open(file, encoding='utf-8') as f:
        for line in f:
            data.append(line.strip())

    return data


#選択肢読み取り
def readChoices(file_name):
    choices=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            line=line.strip()
            choices.append(line.split(' ### '))     #選択肢を区切る文字列

    return choices


#空所内の単語読み取り
def readAns(file_name):
    data=[]
    with open(file_name, encoding='utf-8') as f:
        for line in f:
            line=re.sub(r'.*{ ', '', line)
            line=re.sub(r' }.*', '', line)
            data.append(line.strip())

    return data


#nltkのtoken列をidsへ
def token_to_ids(lang, tokens, maxlen):
    ids=[]
    #NLTKの記号を表すタグ
    symbol_tag=("$", "''", "(", ")", ",", "--", ".", ":", "``", "SYM")
    #NLTKの数詞を表すタグ
    num_tag=("LS", "CD")
    #他のNLTKタグについては nltk.help.upenn_tagset()
    tagged = nltk.pos_tag(tokens)
    for word, tag in tagged:
        if word==CLZ_word:
            ids.append(CLZ_token)
        elif tag in symbol_tag:
            pass
            #記号は無視
        elif tag in num_tag:
            ids.append(NUM_token)
        else:
            ids.append(lang.check_word2index(word.lower()))

    return ids + [PAD_token] * (maxlen - len(ids))


#半角カナとか特殊記号とかを正規化
# Ａ→A，Ⅲ→III，①→1とかそういうの
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


#空所つき1文をidsへ
def sent_to_ids_cloze(lang, s):
    # s は文字列
    s = unicodeToAscii(s)
    s = re.sub(r'[ ]+', ' ', s)
    s = s.strip()
    s = re.sub(r'{.+}', CLZ_word, s)
    token=nltk.word_tokenize(s)
    ids=token_to_ids(lang, token, MAX_LENGTH)

    return ids


#選択肢4つをidsへ
def choices_to_ids(lang, choices):
    # choices は文字列のリスト
    ids=[]
    for choi in choices:
        choi = unicodeToAscii(choi)
        choi = re.sub(r'[ ]+', ' ', choi)
        choi = choi.strip()
        token=nltk.word_tokenize(choi)
        id=token_to_ids(lang, token, C_MAXLEN)
        ids.append(id)

    #デバッグ時確認用
    if len(ids)!=4:
        print('### choices_to_ids ERROR')
        print(choices)
        exit()

    return ids


#正答一つをidsへ
def ans_to_ids(lang, ans, choices):
    # ans は文字列
    # choices は文字列のリスト
    ids = [1 if choi==ans else 0 for choi in choices]

    #デバッグ時確認用
    if sum(ids)!=1:
        print('### ans_to_ids ERROR')
        print(ids)
        print(ans)
        print(choices)
        exit()

    return ids


#fasttextのベクトルファイルから単語辞書とベクトル辞書の作成
def vec_to_dict(vec_path):
    print('Loading fasttext vec ...')
    s=set()
    word_indices=dict()
    indices_word=dict()
    vec_dict=dict()
    i=0
    text=''
    with open(vec_path,'r') as f:
        for line in f:
            if i!=0:
                #先頭行には単語数と次元数が書かれているので無視
                line=line.strip()
                tmp_list=line.split(' ')
                word=tmp_list[0]
                str_list=tmp_list[1:]
                #辞書の作成
                #0番目はパディング用の数字なので使わないことに注意
                word_indices[word]=i
                indices_word[i]=word
                vec_dict[word]=np.array(str_list, dtype=np.float)
            i+=1

    word_indices['#OTHER']=i
    indices_word[i]='#OTHER'
    len_words=i
    return len_words, word_indices, indices_word, vec_dict


#fasttextのベクトルを得る
#未知語の場合にはfasttextのモデル呼び出して実行
#未知語は集合に格納し，あとでファイル出力
def get_ft_vec(word, vec_dict, ft_path, bin_path):
    if word in vec_dict:
        return vec_dict[word]
    elif word in tmp_vec_dict:
        return tmp_vec_dict[word]
    else:
        KeyError_set.add(word)    #要素を追加
        cmd='echo "'+word+'" | '+ft_path+' print-word-vectors '+bin_path
        ret  =  subprocess.check_output(cmd, shell=True)
        line=ret.replace('\n', '').replace('\r','')
        if line[0]==' ':
            line=line[1:]
        if line[-1]==' ':
            line=line[:-1]
        tmp_list=line.split(' ')
        word=tmp_list[0]
        vec=tmp_list[1:]
        vec_array=np.array(vec,dtype=np.float32)
        tmp_vec_dict[word]=vec_array

        return vec_array


###########################
# 2.モデル定義
###########################

# モデルの構築
def build_model(len_words, embedding_matrix):
    f_input=Input(shape=(maxlen_words,))
    f_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(f_input)

    f_full=Dense(vec_size,activation='relu')(f_emb)
    f_layer=LSTM(128)(f_full)

    r_input=Input(shape=(maxlen_words,))
    r_emb=Embedding(output_dim=vec_size, input_dim=len_words+1, input_length=maxlen_words, mask_zero=True, weights=[embedding_matrix], trainable=False)(r_input)

    r_full=Dense(vec_size,activation='relu')(r_emb)
    r_layer=LSTM(128)(r_full)


    merged_layer=add([f_layer, r_layer])

    out_layer=Dense(vec_size,activation='relu')(merged_layer)


    my_model=Model([f_input, r_input], out_layer)

    optimizer = RMSprop()
    my_model.compile(loss='mean_squared_error', optimizer=optimizer)

    return my_model

###########################
# 3.モデルの学習
###########################

#空所等を含まない英文のデータから，モデルの入出力を作成
def make_data(file_path):
    f_X=[]
    r_X=[]
    Y=[]

    #TODO
    #まだ途中



    f=np.array(f_X, dtype=np.int)
    r=np.array(r_X, dtype=np.int)
    y=np.array(Y, dtype=np.float)

    return f, r, y


#checkpoint で保存された最新のモデル(ベストモデルをロード)
def getNewestModel(model):
    files = [(f, os.path.getmtime(f)) for f in glob(save_path+'*hdf5')]
    if len(files) == 0:
        return model
    else:
        newestModel = sorted(files, key=lambda files: files[1])[-1]
        model.load_weights(newestModel[0])
        return model


#学習をn_iters回，残り時間の算出をlossグラフの描画も
def trainIters(model, train_path, val_path, n_iters=5, print_every=10, saveModel=False):

    f_X_train, r_X_train, Y_train=make_data(train_path)
    f_X_val, r_X_val, Y_val=make_data(val_path)


    X_train=[f_X_train, r_X_train]
    X_val=[f_X_val, r_X_val]

    cp_cb = ModelCheckpoint(filepath = save_path+'model_ep{epoch:02d}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

    start = time.time()
    st_time=datetime.datetime.today().strftime('%H:%M')
    print("Training... ", st_time)


    # Ctrl+c で強制終了してもそこまでのモデルで残りの処理継続
    try:
        hist=model.fit(X_train, Y_train, batch_size=BATCH_SIZE, epochs=n_iters, verbose=1, validation_data=(X_val, Y_val), callbacks=[cp_cb], shuffle=True)

        #全学習終わり
        #lossとaccのグラフ描画
        showPlot3(hist.history['loss'], hist.history['val_loss'], 'loss.png', 'loss')

        print(hist.history['loss'])

    except KeyboardInterrupt:
        print()
        print('-' * 89)
        files = [(f, os.path.getmtime(f)) for f in glob(save_path+'*hdf5')]
        if len(files) > 0:
            print('Exiting from training early')
        else :
            exit()

    #ベストモデルのロード
    model=getNewestModel(model)

    return model


#グラフの描画（画像ファイル保存）
def showPlot3(train_plot, val_plot, file_name, label_name):
    fig = plt.figure()
    plt.plot(train_plot, color='blue', marker='o', label='train_'+label_name)
    plt.plot(val_plot, color='green', marker='o', label='val_'+label_name)
    plt.title('model '+label_name)
    plt.xlabel('epoch')
    plt.ylabel(label_name)
    plt.legend()
    plt.savefig(save_path + file_name)


#グラフの描画（画像ファイル保存）
def showPlot4(val_plot, file_name, label_name):
    fig = plt.figure()
    plt.plot(val_plot, color='green', marker='o', label='val_'+label_name)
    plt.title('model '+label_name)
    plt.xlabel('epoch')
    plt.ylabel(label_name)
    plt.legend()
    fig.savefig(save_path + file_name)


###########################
# 4.モデルによる予測
###########################
def model_test(ngram, lang, model, cloze_path, choices_path, ans_path, my_model_kind, data_name='', file_output=True):
    print(data_name)
    use_Ng, use_AR, use_KenLM, use_CAR=use_config(my_model_kind)

    test_X=readCloze(cloze_path)
    test_C=readChoices(choices_path)
    test_Y=readAns(ans_path)

    if args.mode=='mini_test':
        test_X=test_X[:5]
        test_C=test_C[:5]
        test_Y=test_Y[:5]

    X_test_tmp=np.array([sent_to_ids_cloze(lang, s) for s in test_X], dtype=np.int)
    C_test=np.array([choices_to_ids(lang, s) for s in test_C], dtype=np.int)
    Y_test=np.array([ans_to_ids(lang, s, c) for s,c in zip(test_Y, test_C)], dtype=np.bool)

    print('test_ans_rate', np.sum(Y_test, axis=0))

    c1_test, c2_test, c3_test, c4_test = split_choices(C_test)

    # MPALayerの一部: Selective Copying 用の入力
    cloze_test=make_cloze_onehot(X_test_tmp)

    X_test=[X_test_tmp, c1_test, c2_test, c3_test, c4_test, cloze_test]

    # MPALayerの一部: N-gram Statistics 用の入力
    if use_Ng==1:
        N1_test, N2_test, N3_test, N4_test=ngram.get_ngram_count(test_X, test_C)
        X_test.extend([N1_test, N2_test, N3_test, N4_test])

    # 自作拡張: 空所補充文Attentive Reader 用の入力
    if use_CAR==1:
        CAR_sent1_test, CAR_sent2_test, CAR_sent3_test, CAR_sent4_test=ngram.make_CAR_X(lang, test_X, test_C)

        X_test.extend([CAR_sent1_test, CAR_sent2_test, CAR_sent3_test, CAR_sent4_test])


    # 自作拡張: KenLM Score 用の入力
    if use_KenLM==1:
        KenLM_1_test, KenLM_2_test, KenLM_3_test, KenLM_4_test=ngram.get_KenLM_score(test_X, test_C)

        X_test.extend([KenLM_1_test, KenLM_2_test, KenLM_3_test, KenLM_4_test])


    loss, acc=model.evaluate(X_test, Y_test, batch_size=BATCH_SIZE, verbose=1)
    print('loss=%.4f, acc=%.4f' % (loss, acc))

    '''
    #モデルの中間層の出力確認用
    if args.mode=='mini_test':
        layers_names=['concatenate_1', 'concatenate_2', 'concatenate_3', 'concatenate_4', 'concatenate_5']
        for name in layers_names:
            out=Model(inputs=model.input,                               outputs=model.get_layer(name).output).predict(X_test)
            print(name+'\n', out)
    '''

    if file_output:
        with open(save_path+data_name+'_result.txt', 'w') as f:
            f.write('loss=%.4f, acc=%.4f' % (loss, acc))


#コマンドライン引数の設定いろいろ
def get_args():
    parser = argparse.ArgumentParser()
    #miniはプログラムエラーないか確認用的な
    parser.add_argument('--mode', choices=['all', 'mini', 'test', 'mini_test', 'train_loop'], default='all')
    parser.add_argument('--model_dir', help='model directory path (when load model, mode=test)')
    parser.add_argument('--model', help='model file name (when load model, mode=test)')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--model_kind', choices=['origin', 'plus_CAR', 'plus_KenLM', 'plus_both', 'replace_CAR', 'replace_KenLM', 'replace_both'], default='origin', help='model file kind')

    # ほかにも引数必要に応じて追加
    return parser.parse_args()


#----- main部 -----
if __name__ == '__main__':
    #コマンドライン引数読み取り
    args = get_args()
    print(args.mode)
    epoch=args.epoch


    # 2.fasttextのロードと辞書の作成
    '''
    https://github.com/facebookresearch/fastText
    このfastextを事前に実行しておき，その結果を利用
    '''
    ft_path='../../FastText/fastText-0.1.0/fasttext'

    #ベクトルファイル
    vec_path='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.vec'
    bin_path='../../FastText/Model/text8_dim'+str(vec_size)+'_minC0.bin'

    len_words, word_to_id, id_to_word, vec_dict=vec_to_dict(vec_path)

    weights_matrix = np.zeros((len_words+1, vec_size))
    if args.mode == 'all':
        for i in range(len_words):
            if i!=0:
                #IDが0の単語が存在しないので0は飛ばす
                weights_matrix[i] = get_ft_vec(id_to_word[i], vec_dict, ft_path, bin_path)

    #通常時
    # 2.モデル定義
    model = build_model(len_words, weights_matrix)

    #学習時
    if args.mode == 'all' or args.mode == 'mini':
        train_path=CLOTH_path+'for_KenLM_CLOTH.txt'
        val_path=CLOTH_path+'for_KenLM_CLOTH_val.txt'

        #モデルとか結果とかを格納するディレクトリの作成
        save_path=save_path+args.mode+'_BiVecPresModel'
        if os.path.exists(save_path)==False:
            os.mkdir(save_path)
        save_path=save_path+'/'
        plot_model(model, to_file=save_path+'model_'+args.model_kind+'.png', show_shapes=True)
        #model.summary()

        # 3.学習
        model = trainIters(model, train_data, val_data, n_iters=epoch, saveModel=True)
        print('Train end')

    #すでにあるモデルでテスト時
    else:
        save_path=args.model_dir+'/'
        model.load_weights(save_path+args.model+'.hdf5')
        #model.summary()

        save_path=save_path+today_str

    # 4.評価
    center_cloze=git_data_path+'center_cloze.txt'
    center_choi=git_data_path+'center_choices.txt'
    center_ans=git_data_path+'center_ans.txt'

    MS_cloze=git_data_path+'microsoft_cloze.txt'
    MS_choi=git_data_path+'microsoft_choices_for_CLOTH.txt' #これ4択のテストデータ
    MS_ans=git_data_path+'microsoft_ans.txt'

    high_path=git_data_path+'CLOTH_test_high'
    middle_path=git_data_path+'CLOTH_test_middle'

    CLOTH_high_cloze = high_path+'_cloze.txt'
    CLOTH_high_choi = high_path+'_choices.txt'
    CLOTH_high_ans = high_path+'_ans.txt'

    CLOTH_middle_cloze = middle_path+'_cloze.txt'
    CLOTH_middle_choi = middle_path+'_choices.txt'
    CLOTH_middle_ans = middle_path+'_ans.txt'

    is_out=False    #ファイル出力一括設定用

    model_test(clothNg, vocab, model, center_cloze, center_choi, center_ans, my_model_kind, data_name='center', file_output=is_out)

    if args.mode != 'mini' and args.mode != 'mini_test':
        model_test(clothNg, vocab, model, MS_cloze, MS_choi, MS_ans, data_name='MS', file_output=is_out)

        model_test(clothNg, vocab, model, CLOTH_high_cloze, CLOTH_high_choi, CLOTH_high_ans, my_model_kind, data_name='CLOTH_high', file_output=is_out)

        model_test(clothNg, vocab, model, CLOTH_middle_cloze, CLOTH_middle_choi, CLOTH_middle_ans, my_model_kind, data_name='CLOTH_middle', file_output=is_out)
