# -*- coding: utf-8 -*-

'''
空所の位置ごとにBLEUとかaccとか計算
'''

from __future__ import print_function

import collections
import math
import codecs
import os
import re
import subprocess

import tensorflow as tf



#----- グローバル変数一覧 -----




#----- とってきた関数群 -----



def _get_ngrams(segment, max_order):
  """Extracts all n-grams upto a given maximum order from an input segment.
  Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.
  Returns:
    The Counter containing all n-grams upto max_order in segment
    with a count of how many times each n-gram occurred.
  """
  ngram_counts = collections.Counter()
  for order in range(1, max_order + 1):
    for i in range(0, len(segment) - order + 1):
      ngram = tuple(segment[i:i+order])
      ngram_counts[ngram] += 1
  return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4,
                 smooth=False):
  """Computes BLEU score of translated segments against one or more references.
  Args:
    reference_corpus: list of lists of references for each translation. Each
        reference should be tokenized into a list of tokens.
    translation_corpus: list of translations to score. Each translation
        should be tokenized into a list of tokens.
    max_order: Maximum n-gram order to use when computing BLEU score.
    smooth: Whether or not to apply Lin et al. 2004 smoothing.
  Returns:
    3-Tuple with the BLEU score, n-gram precisions, geometric mean of n-gram
    precisions and brevity penalty.
  """
  matches_by_order = [0] * max_order
  possible_matches_by_order = [0] * max_order
  reference_length = 0
  translation_length = 0
  for (references, translation) in zip(reference_corpus,
                                       translation_corpus):
    reference_length += min(len(r) for r in references)
    translation_length += len(translation)

    merged_ref_ngram_counts = collections.Counter()
    for reference in references:
      merged_ref_ngram_counts |= _get_ngrams(reference, max_order)
    translation_ngram_counts = _get_ngrams(translation, max_order)
    overlap = translation_ngram_counts & merged_ref_ngram_counts
    for ngram in overlap:
      matches_by_order[len(ngram)-1] += overlap[ngram]
    for order in range(1, max_order+1):
      possible_matches = len(translation) - order + 1
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

  if reference_length!=0:
    ratio = float(translation_length) / reference_length
    if ratio > 1.0:
        bp = 1.
    else:
        bp = math.exp(1 - 1. / ratio)
    bleu = geo_mean * bp
  else:
    ratio=0
    bp=0
    bleu=0


  return (bleu, precisions, bp, ratio, translation_length, reference_length)




def my_clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  #sentence = sentence.strip()   #空白と改行を削除？
  sentence=re.sub(r'[^a-z{}<> ]', '', sentence)

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence



def bleu(ref_file, trans_file, subword_option=None):
  """Compute BLEU scores and handling BPE."""
  max_order = 4
  smooth = False

  ref_files = [ref_file]
  reference_text = []
  for reference_filename in ref_files:
    with codecs.getreader("utf-8")(
        tf.gfile.GFile(reference_filename, "rb")) as fh:
      reference_text.append(fh.readlines())

  per_segment_references = []
  for references in zip(*reference_text):
    reference_list = []
    for reference in references:
      reference = my_clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = my_clean(line, subword_option=None)
      translations.append(line.split(" "))

  # bleu_score, precisions, bp, ratio, translation_length, reference_length
  bleu_score, _, _, _, _, _ = compute_bleu(
      per_segment_references, translations, max_order, smooth)
  return 100 * bleu_score




#----- 自作関数 -----

def is_correct_cloze(line):
    left=line.count('{')
    right=line.count('}')
    if left*right==1:
        return True
    elif left+right>1:
        #print(line)
        pass
    else:
        #print(line)
        pass
    return False


def get_cloze(ref, ans):
    ref=re.sub(r'.*{ ', '', ref)
    ref_cloze=re.sub(r' }.*', '', ref)

    ans=re.sub(r'.*{ ', '', ans)
    ans_cloze=re.sub(r' }.*', '', ans)

    return ref_cloze, ans_cloze



def get_ans_length(ans_cloze):
    ans_li=ans_cloze.split(' ')
    ans_len=len(ans_li)

    return ans_len

def match(ref_cloze, ans_cloze):
    ref_set=set(ref_cloze.split(' '))
    ans_set=set(ans_cloze.split(' '))
    i=0
    '''
    print(ref_set)
    print(ans_set)
    print()
    '''
    for word in ref_set:
        if word in ans_set:
            i+=1

    return i



def calc_acc_and_partacc(ref_path, ans_path):
    ct=0
    line_num=0
    match_line_num=0
    with open(ref_path) as f_ref:
        with open(ans_path) as f_ans:
            for ref_line in f_ref:
                ans_line=f_ans.readline()
                ref_line=re.sub(r'[^a-z{}<> ]', '', ref_line)
                ans_line=re.sub(r'[^a-z{}<> ]', '', ans_line)
                ref_cloze,ans_cloze=get_cloze(ref_line, ans_line)
                tmp_ans_length=get_ans_length(ans_cloze)
                line_num+=1
                if is_correct_cloze(ref_line):
                    tmp_match=match(ref_cloze, ans_cloze)
                    if tmp_match > 0:
                        match_line_num+=1
                    if tmp_ans_length == tmp_match:
                        ct+=1

    print(line_num)
    print(match_line_num)

    print(ct)
    if line_num==0:
        acc=0
        part_acc=0
    else:
        acc=1.0*ct/line_num
        part_acc=1.0*match_line_num/line_num
    return acc, part_acc


def print_result(ref_path, ans_path):
    print('file: ',ref_path)
    BLEU=bleu(ref_path, ans_path)
    acc, part_acc = calc_acc_and_partacc(ref_path, ans_path)

    print('BLEU:     ', '{0:.2f}'.format(BLEU))
    print('acc:      ', '{0:.2f}'.format(acc))
    print('part_acc: ', '{0:.2f}'.format(part_acc))


def rename(file_path):
    head='/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/tmp/'
    tmp=file_path.rfind('/')
    file_path=file_path[tmp+1:]
    file_path=file_path[:-4]

    return head + file_path


def devide_file_and_print(ref_path, ans_path):
    top_ref=rename(ref_path) + '_top.txt'
    top_ans=rename(ans_path) + '_top.txt'
    end_ref=rename(ref_path) + '_end.txt'
    end_ans=rename(ans_path) + '_end.txt'
    other_ref=rename(ref_path) + '_other.txt'
    other_ans=rename(ans_path) + '_other.txt'

    with open(ref_path) as f_ref:
        with open(ans_path) as f_ans:
            f_ref_top = open(top_ref, 'w')
            f_ans_top = open(top_ans, 'w')
            f_ref_end = open(end_ref, 'w')
            f_ans_end = open(end_ans, 'w')
            f_ref_other=open(other_ref, 'w')
            f_ans_other=open(other_ans, 'w')
            for ref_line in f_ref:
                ans_line=f_ans.readline()
                if ans_line[0]=='{':
                    f_ref_top.write(ref_line)
                    f_ans_top.write(ans_line)
                elif ans_line[-2]=='}':
                    f_ref_end.write(ref_line)
                    f_ans_end.write(ans_line)
                else:
                    f_ref_other.write(ref_line)
                    f_ans_other.write(ans_line)

            f_ref_top.close()
            f_ans_top.close()
            f_ref_end.close()
            f_ans_end.close()
            f_ref_other.close()
            f_ans_other.close()


    print_result(top_ref, top_ans)
    print_result(end_ref, end_ans)
    print_result(other_ref, other_ans)


#return なし


#----- いわゆるmain部みたいなの -----
dev_predict='/home/tamaki/M2/Tensorflow/nmt/text8_output_ep100_2/output_dev.txt'
dev_ans='/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt_dev.ans'

test_predict='/home/tamaki/M2/Tensorflow/nmt/text8_output_ep100_2/output_infer.txt'
test_ans='/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/center_nmt.ans'



devide_file_and_print(dev_predict, dev_ans)
print('\n\n')

devide_file_and_print(test_predict, test_ans)
