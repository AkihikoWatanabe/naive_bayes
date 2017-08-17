# coding=utf-8

import numpy as np
cimport numpy as np # noqa

DTYPE_INT32 = np.int32
DTYPE_FLOAT32 = np.float32
ctypedef np.int32_t DTYPE_INT32_t
ctypedef np.float32_t DTYPE_FLOAT32_t

cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def multivariate_bernoulli_scores(
        np.ndarray[DTYPE_INT32_t, ndim=1] words,
        np.ndarray[DTYPE_INT32_t, ndim=1] cats,
        np.ndarray[DTYPE_INT32_t, ndim=1] vocab,
        dict pc,
        dict pwc):
    """ 与えられた文書の多変数ベルヌーイモデルにおける各ラベルに対するスコア計算を行う

    Args:
        words (np.ndarray): 1文書の単語リスト
        cats (np.ndarray): ラベル一覧
        vocab (np.ndarray): 計算に用いる語彙
        pc (dict): ラベル生起確率
        pwc (dict): ラベルごとの単語生起確率

    Returns:
        dict: labelをkey, scoreをvalueとして持つdict
    """

    cdef int N = len(cats)
    cdef dict label_scores = {}
    cdef float score = 0.0

    for i, cat in enumerate(cats):
        score += np.log(pc[cat])
        for w in vocab:
            try:
                score += multivariate_bernoulli_word_prob(
                        cat, w, words, pwc)
            except KeyError:
                score = float("-inf")
                break
        label_scores[cat] = score
        score = 0.0

    return label_scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def multivariate_bernoulli_word_prob(
        int cat,
        int w,
        np.ndarray[DTYPE_INT32_t, ndim=1] words,
        dict pwc):
    """ 多変数ベルヌーイモデルのスコア計算において、単語のスコアを計算する。

    Args:
        cat (int): 計算対象のラベル
        w (int): 計算対象の単語
        words (np.ndarray): 文書の単語リスト
        pwc (dict): カテゴリごとの単語生起確率
    """

    if w in words:
        return np.log(pwc[cat][w])
    if w in pwc[cat]:
        return np.log(1.0 - pwc[cat][w])
    else:
        return 0.0


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def multinomial_scores(
        np.ndarray[DTYPE_INT32_t, ndim=1] cats,
        dict cnt,
        dict pc,
        dict pwc):
    """ 与えられた文書の多項モデルにおける各ラベルに対するスコア計算を行う

    Args:
        cats (np.ndarray): ラベル一覧
        cnt (dict): 単語をkey, 単語の出現回数をvalueとして持つdict
        pc (dict): ラベル生起確率
        pwc (dict): ラベルごとの単語生起確率

    Returns:
        dict: labelをkey, scoreをvalueとして持つdict
    """

    cdef int N = len(cats)
    cdef dict label_scores = {}
    cdef float score = 0.0

    for i, cat in enumerate(cats):
        score += np.log(pc[cat])
        for w, c in cnt.items():
            try:
                score += multinomial_word_prob(cat, w, c, pwc)
            except:
                score = float("-inf")
                break

        label_scores[cat] = score
        score = 0.0

    return label_scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def multinomial_word_prob(int cat, int w, int c, dict pwc):
    """ 多項モデルのスコア計算において、単語のスコアを計算する。

    Args:
        cat (int): 計算対象のラベル
        w (int): 計算対象の単語
        c (int): 計算対象の単語の出現回数
        pwc (dict): カテゴリごとの単語生起確率
    """

    if pwc[cat][w] == 0.0:
        raise Exception

    return c * np.log(pwc[cat][w])
