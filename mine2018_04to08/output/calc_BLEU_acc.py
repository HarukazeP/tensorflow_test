# -*- coding: utf-8 -*-

'''
seq2seqの結果からBLEUとかaccとか計算
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

  ratio = float(translation_length) / reference_length

  if ratio > 1.0:
    bp = 1.
  else:
    bp = math.exp(1 - 1. / ratio)

  bleu = geo_mean * bp

  return (bleu, precisions, bp, ratio, translation_length, reference_length)




def _clean(sentence, subword_option):
  """Clean and handle BPE or SPM outputs."""
  sentence = sentence.strip()

  # BPE
  if subword_option == "bpe":
    sentence = re.sub("@@ ", "", sentence)

  # SPM
  elif subword_option == "spm":
    sentence = u"".join(sentence.split()).replace(u"\u2581", u" ").lstrip()

  return sentence



def _bleu(ref_file, trans_file, subword_option=None):
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
      reference = _clean(reference, subword_option)
      reference_list.append(reference.split(" "))
    per_segment_references.append(reference_list)

  translations = []
  with codecs.getreader("utf-8")(tf.gfile.GFile(trans_file, "rb")) as fh:
    for line in fh:
      line = _clean(line, subword_option=None)
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
        print(line)
    return False



def check_cloze(ref, ans):
    ref=re.sub(r'.*{', '', ref)
    ref=re.sub(r'}.*', '', ref)
    
    ans=re.sub(r'.*{', '', ans)
    ans=re.sub(r'}.*', '', ans)
    
    return ref==ans


def check_sent(ref, ans):
    ref=re.sub(r'{.*}', '', ref)
    ans=re.sub(r'{.*}', '', ans)
    
    return ref==ans


def count_correct_num(ref_path, ans_path):
    cloze_sent_num=0
    all_correct_num=0
    cloze_correct_num=0
    sent_correct_num=0
    line_num=0
    with open(ref_path) as f_ref:
        with open(ans_path, 'w') as f_ans:
            for ref_line in f_ref:
                line_num+=1
                ans_line=f_ans.readline()
                if is_correct_cloze(ref_line):
                    cloze_sent_num+=1
                    ref_line=re.sub(r'[^a-z{}<> ]', '', ref_line)
                    ans_line=re.sub(r'[^a-z{}<> ]', '', ans_line)
                    if ref_line == ans_line:
                        all_correct_num+=1
                        cloze_correct_num+=1
                        sent_correct_num+=1
                    else:
                        if check_cloze(ref_line, ans_line):
                            cloze_correct_num+=1
                        if check_sent(ref_line, ans_line):
                            sent_correct_num+=1


    '''
    返り値は
    line_num：問題文の数
    cloze_sent_num：{と}の両方を含む文の数
    all_correct_num：予測と教師データが完全一致している数
    cloze_correct_num：空所内で予測と教師データが完全一致している数
    sent_correct_num：空所内で予測と教師データが完全一致している数
    
    '''
    return line_num, cloze_sent_num, all_correct_num, cloze_correct_num, sent_correct_num



def print_result(ref_path, ans_path):
    print('file: ',ref_path)
    BLEU=_bleu(ref_path, ans_path)
    line, num_cloze, all, cloze, sent =count_correct_num(ref_path, ans_path)
    print('file: ',ref_path)
    print('BLEU: ',BLEU)
    print('acc(all): ',1.0*all/line)
    print('acc(cloze): ',1.0*cloze/line)
    print('acc(sent): ',1.0*sent/line)
    print('num: ',line)
    print('cloze miss: ',line - num_cloze)

#return なし


#----- いわゆるmain部みたいなの -----


print_result('output_dev.txt','/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/text8_nmt_dev.ans')

print_result('output_infer.txt','/home/tamaki/M2/Tensorflow/mine2018_4to7/Data/my_nmt/center_nmt.ans')


