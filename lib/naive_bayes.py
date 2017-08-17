# coding=utf-8

"""
ナイーブベイズ分類器を提供するモジュール
"""

from collections import Counter
from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
from abc import ABCMeta, abstractmethod

from .serializer import Serializer
from .const import (
        NBParam,
        LEARNING_MLE,
        LEARNING_MAP,
        FS_NO,
        FS_FREQ,
        FS_PMI
)
from .vocab import Vocab
from .calc_score.calc_score_naive_bayes import (
        multivariate_bernoulli_scores as mb_scores,
        multinomial_scores as mnmal_scores
)


class NaiveBayes(metaclass=ABCMeta):
    """ ナイーブベイズ分類器の抽象クラス

    Attributes:
        pc (dict): パラメータ（カテゴリの生起確率）
        pwc (dict): パラメータ（カテゴリごとの単語の生起確率）
        wvocab (Vocab): テキストの単語情報を格納するVocab
        cvocab (Vocab): 分類ラベルを格納するVocab
        learning (str): 学習方法（MLE or MAP）
        ftr_slct (str): 素性選択方法 (FREQ or PMI)
        stopwords_num (int): ストップワードの数
        stopwords (list): ストップワードリスト, ストップワードの表層を要素として持つリスト
        vocab_size (int): 語彙のサイズ
        alpha (int): ディリクレ分布のハイパーパラメータ（MAP推定で用いる, default: 1）
        　　　　　　 単語の出現回数にどれだけ下駄を履かせるかに相当
    """

    def __init__(self, param, use_cython=False):
        """ 学習方法、パラメータ、語彙の初期化を行う。

        Args:
            param (NBParam): ナイーブベイズ分類器で用いるパラメータを格納したnamedtuple
            use_cython (bool): Cythonモジュールを使用するか否か
        """

        self.pc = {}
        self.pwc = {}
        self.wvocab = Vocab(param.vocab_size)
        self.learning = param.learning
        self.ftr_slct = param.ftr_slct
        self.stopwords_num = param.stopwords_num
        self.alpha = param.alpha
        self.use_cython = use_cython

    def save(self, file_path):
        """ 学習したモデルを保存する。

        Args:
            file_path (str): モデルの保存先
        """

        save_data = [
                self.pc,
                self.pwc,
                dict(self.wvocab.w2i),
                self.wvocab.i2w,
                self.wvocab.size,
                dict(self.cvocab.w2i),
                self.cvocab.i2w,
                self.cvocab.size,
                self.learning,
                self.ftr_slct,
                self.stopwords_num,
                self.stopwords,
                self.use_cython,
        ]
        Serializer.dump_data(save_data, file_path)

    @staticmethod
    def load(file_path):
        """ 学習したモデルを読み込んで返す。

        Args:
            file_path (str): モデルの読み込み元

        Returns:
            model: 読み込んだモデルのインスタンス
        """

        self = MultivariateBernoulli(
                NBParam(
                    None,
                    None,
                    0,
                    0,
                    None
                )
        )
        (pc, pwc, w_w2i, w_i2w, w_vsize,
            c_w2i, c_i2w, c_vsize, learning, ftr_slct,
            snum, swords, use_cython) = Serializer.load_data(file_path)
        self.pc = pc
        self.pwc = pwc
        self.wvocab = Vocab(w_vsize)
        self.wvocab.set_w2i(w_w2i)
        self.wvocab.set_i2w(w_i2w)
        self.cvocab = Vocab(c_vsize)
        self.cvocab.set_w2i(c_w2i)
        self.cvocab.set_i2w(c_i2w)
        self.learning = learning
        self.ftr_slct = ftr_slct
        self.stopwords_num = snum
        self.stopwords = swords
        self.use_cython = use_cython

        return self

    def learn(self, train):
        """ パラメータの学習を行う

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                          単語リストはuniqされていることが前提
        """

        if self.learning == LEARNING_MLE:
            self.MLE(train)
        elif self.learning == LEARNING_MAP:
            self.MAP(train)

    def make_stopwords_vocab(self, train):
        """ ストップワードリスト、および語彙を作成する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        # make stopwords
        global_cnt = Counter()
        for cat, docs in train.items():
            w_flatten = [w for words in docs for w in words]
            cnt = Counter(w_flatten)
            global_cnt += cnt
        self.stopwords = [
                w for (w, c) in global_cnt.most_common()[:self.stopwords_num]
        ]

        # make vocab
        if self.ftr_slct == FS_NO:
            [self.wvocab.add(w)
                for (w, _) in global_cnt.items()]
        elif self.ftr_slct == FS_FREQ:
            frequent_words = global_cnt.most_common()[self.stopwords_num:]
            [self.wvocab.add(w)
                for (w, _) in frequent_words[:self.wvocab.size]]
        elif self.ftr_slct == FS_PMI:
            pmi = self.PMI(train)
            flatten = [
                    (w, v) for _, _pmi in pmi.items() for w, v in _pmi.items()
            ]
            flatten = sorted(flatten, key=lambda x: x[1], reverse=True)
            [self.wvocab.add(w) for (w, _) in flatten
                if not self.wvocab.is_full()]

        self.cvocab = Vocab(len(train.keys()))
        [self.cvocab.add(cat) for cat in train.keys()]

    def PMI(self, train):
        """ PMI(cateogory, word)を計算して返す。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        pmi, pc, pw, pwc = {}, {}, {}, {}
        N = sum([len(docs) for docs in train.values()])
        global_uniq_cnt = Counter()
        for cat, docs in train.items():
            pc[cat] = len(docs) / N
            uniq_cnt = Counter()
            for words in docs:
                uniq_cnt += Counter(list(set(words)))
            pwc[cat] = {w: c / N for w, c in uniq_cnt.items()}

            global_uniq_cnt += uniq_cnt
        pw = {w: c / N for w, c in global_uniq_cnt.items()}

        for cat in train.keys():
            pmi[cat] = {
                    w: np.log(pwc[cat][w] / (pc[cat] * pw[w]))
                    if w in pwc[cat] else float("-inf")
                    for w in pw.keys()
            }

        return pmi

    @abstractmethod
    def MLE(self, train):
        """ 最尤推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        raise NotImplementedError()

    @abstractmethod
    def MAP(self, train):
        """ MAP推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        raise NotImplementedError()

    @abstractmethod
    def predict(self, test, get_score=False):
        """ 学習したパラメータを用いて、テキストのカテゴリを分類する。

        Args:
            test (np.array): 予測を行う各文書の単語リストを要素として持つリスト
            get_score (bool): 予測ラベルのスコアを返すか否か

        Returns:
            list: 予測結果のラベル系列
            float: 予測ラベルのスコア(get_scoreがTrueの時のみ)
        """

        raise NotImplementedError()


class MultivariateBernoulli(NaiveBayes):
    """ 多変数ベルヌーイモデルを実装したクラス

    Attributes:
        pc (dict): パラメータ（カテゴリの生起確率）
        pwc (dict): パラメータ（カテゴリごとの単語の生起確率）
        wvocab (Vocab): テキストの単語情報を格納するVocab
        cvocab (Vocab): 分類ラベルを格納するVocab
        learning (str): 学習方法（MLE or MAP）
        ftr_slct (str): 素性選択方法 (FREQ or PMI)
        stopwords_num (int): ストップワードの数
        stopwords (list): ストップワードリスト, ストップワードの表層を要素として持つリスト
        vocab_size (int): 語彙のサイズ
        alpha (int): ディリクレ分布のハイパーパラメータ（MAP推定で用いる, default: 1）
        　　　　　　 単語の出現回数にどれだけ下駄を履かせるかに相当
    """

    def MLE(self, train):
        """ 最尤推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        self.make_stopwords_vocab(train)

        # パラメータの学習
        N = sum([len(docs) for docs in train.values()])
        for cat, docs in tqdm(train.items()):
            catid = self.cvocab.w2i[cat]
            cnt = Counter(
                    [w for words in docs for w in words
                        if w in self.wvocab.w2i]
            )
            self.pwc[catid] = {
                    self.wvocab.w2i[w]: c / len(docs) for w, c in cnt.items()
            }
            self.pc[catid] = len(docs) / N

    def MAP(self, train):
        """ MAP推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提
        """

        self.make_stopwords_vocab(train)

        # パラメータの学習
        N = sum([len(docs) for docs in train.values()])
        cat_num = len(train.keys())
        for cat, docs in tqdm(train.items()):
            catid = self.cvocab.w2i[cat]
            cnt = Counter(
                    [w for words in docs for w in words
                        if w in self.wvocab.w2i]
            )
            self.pwc[catid] = {
                    self.wvocab.w2i[w]: (cnt[w] + (self.alpha - 1)) /
                    (len(docs) + 2 * (self.alpha - 1))
                    for w in self.wvocab.w2i.keys()
            }
            self.pc[catid] = (len(docs) + (self.alpha - 1)) / (
                    N + cat_num * (self.alpha - 1))

    def predict(self, test, get_score=False):
        """ 学習したパラメータを用いて、テキストのカテゴリを分類する。

        Args:
            test (np.array): 予測を行う各文書の単語リストを要素として持つリスト
            get_score (bool): 予測ラベルのスコアを返すか否か

        Returns:
            list: 予測結果のラベル系列
            float: 予測ラベルのスコア(get_scoreがTrueの時のみ)
        """

        SCORE_IDX = 1
        catid_scores = []
        for words in tqdm(test):
            wordids = np.asarray(
                    [self.wvocab.w2i[w] for w in words
                        if w in self.wvocab.w2i],
                    dtype=np.int32
            )
            catids = np.asarray(
                    list(self.cvocab.w2i.values()),
                    dtype=np.int32
            )
            if self.use_cython:
                vocab = np.asarray(
                        list(self.wvocab.w2i.values()),
                        dtype=np.int32
                )
                _catid_scores = mb_scores(
                        wordids,
                        catids,
                        vocab,
                        self.pc,
                        self.pwc
                )
                _catid_scores = _catid_scores.items()
            else:
                _catid_scores = self.__scores(wordids)
            catid_scores.append(
                    max(_catid_scores, key=lambda x: x[SCORE_IDX])
            )

        if get_score:
            return [(self.cvocab.i2w[catid], score)
                    for (catid, score) in catid_scores]
        else:
            return [self.cvocab.i2w[catid] for (catid, score) in catid_scores]

    def __scores(self, wordids):
        """ 与えられた単語リストから、各カテゴリごとにスコアを計算し返す。

        Args:
            words (np.ndarray): 単語リスト

        Returns:
            list: (label, score) を要素として持つリスト
        """

        label_scores = []
        for catid in self.pc.keys():
            score = 0.0
            score += np.log(self.pc[catid])
            try:
                score += sum(
                        [self.__word_prob(catid, wid, wordids)
                            for wid in self.wvocab.w2i.values()]
                )
            except KeyError:
                score = float("-inf")
            label_scores.append((catid, score))

        return label_scores

    def __word_prob(self, catid, wid, wordids):
        """ 単語の生起確率を計算する。

        Args:
            cat (int): カテゴリ
            wid (int): 生起確率を計算する単語
            wordids (list): 文書の単語リスト

        Returns:
            float: widの生起確率
        """

        if wid in wordids:
            return np.log(self.pwc[catid][wid])
        else:
            try:
                return np.log(1.0 - self.pwc[catid][wid])
            except KeyError:
                return 0.0


class Multinomial(NaiveBayes):
    """ 多項モデルを実装したクラス

    Attributes:
        pc (dict): パラメータ（カテゴリの生起確率）
        pwc (dict): パラメータ（カテゴリごとの単語の生起確率）
        wvocab (Vocab): テキストの単語情報を格納するVocab
        cvocab (Vocab): 分類ラベルを格納するVocab
        learning (str): 学習方法（MLE or MAP）
        ftr_slct (str): 素性選択方法 (FREQ or PMI)
        stopwords_num (int): ストップワードの数
        stopwords (list): ストップワードリスト, ストップワードの表層を要素として持つリスト
        vocab_size (int): 語彙のサイズ
        alpha (int): ディリクレ分布のハイパーパラメータ（MAP推定で用いる, default: 1）
        　　　　　　 単語の出現回数にどれだけ下駄を履かせるかに相当
    """

    def MLE(self, train):
        """ 最尤推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていない
        """

        self.make_stopwords_vocab(train)

        # パラメータの学習
        N = sum([len(docs) for docs in train.values()])
        for cat, docs in tqdm(train.items()):
            catid = self.cvocab.w2i[cat]
            w_flatten = [
                    w for words in docs for w in words
                    if w in self.wvocab.w2i
            ]
            cnt = Counter(w_flatten)
            self.pwc[catid] = {
                    self.wvocab.w2i[w]: cnt[w] / len(w_flatten)
                    for w in self.wvocab.w2i.keys()
            }
            self.pc[catid] = len(docs) / N

    def MAP(self, train):
        """ MAP推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていない
        """

        self.make_stopwords_vocab(train)

        # パラメータの学習
        N = sum([len(docs) for docs in train.values()])
        cat_num = len(train.keys())
        for cat, docs in tqdm(train.items()):
            catid = self.cvocab.w2i[cat]
            w_flatten = [
                    w for words in docs for w in words
                    if w in self.wvocab.w2i
            ]
            cnt = Counter(w_flatten)
            self.pwc[catid] = {
                    self.wvocab.w2i[w]: (cnt[w] + (self.alpha - 1)) /
                    (len(w_flatten) + len(self.wvocab.w2i) * (self.alpha - 1))
                    for w in self.wvocab.w2i.keys()
            }
            self.pc[catid] = (len(docs) + (self.alpha - 1)) / (
                    N + cat_num * (self.alpha - 1))

    def predict(self, test, get_score=False):
        """ 学習したパラメータを用いて、テキストのカテゴリを分類する。

        Args:
            test (np.array): 予測を行う各文書の単語リストを要素として持つリスト
            get_score (bool): 予測ラベルのスコアを返すか否か

        Returns:
            list: 予測結果のラベル系列
            float: 予測ラベルのスコア(get_scoreがTrueの時のみ)
        """

        SCORE_IDX = 1
        catid_scores = []
        for words in tqdm(test):
            wordids = np.asarray(
                    [self.wvocab.w2i[w] for w in words
                        if w in self.wvocab.w2i],
                    dtype=np.int32
            )
            catids = np.asarray(
                    list(self.cvocab.w2i.values()),
                    dtype=np.int32
            )
            if self.use_cython:
                _catid_scores = mnmal_scores(
                        catids,
                        dict(Counter(wordids)),
                        self.pc,
                        self.pwc
                )
                _catid_scores = _catid_scores.items()
            else:
                _catid_scores = self.__scores(wordids)
            catid_scores.append(
                    max(_catid_scores, key=lambda x: x[SCORE_IDX])
            )

        if get_score:
            return [(self.cvocab.i2w[catid], score)
                    for (catid, score) in catid_scores]
        else:
            return [self.cvocab.i2w[catid] for (catid, score) in catid_scores]

    def __scores(self, wordids):
        """ 与えられた単語リストから、各カテゴリごとにスコアを計算し返す。

        Args:
            wordids (list): 単語リスト

        Returns:
            list: (label, score) を要素として持つリスト
        """

        label_scores = []
        cnt = Counter(wordids)
        for catid in self.cvocab.w2i.values():
            score = 0.0
            score += np.log(self.pc[catid])
            try:
                score += sum(
                        [self.__word_prob(catid, wid, c)
                            for (wid, c) in cnt.items()]
                )
            except:
                score = float("-inf")
            label_scores.append((catid, score))

        return label_scores

    def __word_prob(self, catid, wid, c):
        """ 単語の生起確率を計算する。

        Args:
            catid (int): カテゴリ
            wid (int): 生起確率を計算する単語
            c (int): target_wの文書内での出現回数

        Returns:
            float: target_wの生起確率
        """

        if self.pwc[catid][wid] == 0.0:
            raise Exception

        return c * np.log(self.pwc[catid][wid])


class TWCNB(NaiveBayes):
    """ Transformed Weight-normalized Complement Naive Bayesを実装したクラス

    Paper:
        Tackling the Poor Assumptions of Naive Bayes Text Classifiers,
        Rennie+, ICML'03
        http://machinelearning.wustl.edu/mlpapers/paper_files/icml2003_RennieSTK03.pdf

    Attributes:
        pwc (dict): パラメータ（Complement-Categoryに基づいた単語のスコア）
        cat2scores (dict): 各カテゴリの単語頻度にtransformationを加えたスコア
        wvocab (Vocab): テキストの単語情報を格納するVocab
        cvocab (Vocab): 分類ラベルを格納するVocab
        learning (str): 学習方法（MLE or MAP）
        stopwords_num (int): ストップワードの数
        stopwords (list): ストップワードリスト, ストップワードの表層を要素として持つリスト
        vocab_size (int): 語彙のサイズ
        alpha (int): ディリクレ分布のハイパーパラメータ（MAP推定で用いる, default: 1）
        　　　　　　 単語の出現回数にどれだけ下駄を履かせるかに相当
    """

    def __init__(self, param, use_cython=False):
        """ 学習方法、パラメータ、語彙の初期化を行う。

        Args:
            param (NBParam): ナイーブベイズ分類器で用いるパラメータを格納したnamedtuple
        """

        super(TWCNB, self).__init__(param, use_cython=use_cython)
        self.cat2scores = {}  # for debug

    def save(self, file_path):
        """ 学習したモデルを保存する。

        Args:
            file_path (str): モデルの保存先
        """

        save_data = [
                self.pwc,
                dict(self.wvocab.w2i),
                self.wvocab.i2w,
                self.wvocab.size,
                dict(self.cvocab.w2i),
                self.cvocab.i2w,
                self.cvocab.size,
                self.learning,
                self.ftr_slct,
                self.stopwords_num,
                self.stopwords,
                self.use_cython,
                self.idf
        ]
        Serializer.dump_data(save_data, file_path)

    @staticmethod
    def load(file_path):
        """ 学習したモデルを読み込んで返す。

        Args:
            file_path (str): モデルの読み込み元

        Returns:
            TWCNB: 読み込んだモデルのインスタンス
        """

        self = TWCNB(
                NBParam(
                    None,
                    None,
                    0,
                    0,
                    None
                )
        )
        (pwc, w_w2i, w_i2w, w_vsize,
            c_w2i, c_i2w, c_vsize, learning, ftr_slct, snum,
            swords, use_cython, idf) = Serializer.load_data(file_path)
        self.pwc = pwc
        self.wvocab = Vocab(w_vsize)
        self.wvocab.set_w2i(w_w2i)
        self.wvocab.set_i2w(w_i2w)
        self.cvocab = Vocab(c_vsize)
        self.cvocab.set_w2i(c_w2i)
        self.cvocab.set_i2w(c_i2w)
        self.learning = learning
        self.ftr_slct = ftr_slct
        self.stopwords_num = snum
        self.stopwords = swords
        self.use_cython = use_cython
        self.idf = idf

        return self

    def MLE(self, train):
        """ TWCNBではMAPのみを実装しているため、MAPを呼び出す。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていない
        """

        print("Warning: Original paper used MAP estimation only.")
        print("Warning: Hence, we use MAP estimation instead of MLE.")

        self.MAP(train)

    def MAP(self, train):
        """ MAP推定を用いてパラメータを学習する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていない
        """

        self.make_stopwords_vocab(train)
        self.__make_idf(train)
        cat2scores = self.__count_words_by_category(train)
        self.cat2scores = cat2scores

        for catid in tqdm(self.cvocab.w2i.values()):
            ccat_scores = self.__count_words_by_complement_category(
                    catid,
                    cat2scores
            )
            weights = (ccat_scores.toarray() + (self.alpha - 1)) / (
                    ccat_scores.sum() + len(self.wvocab.w2i) * (self.alpha - 1)
            )
            weights = np.log(weights)
            weights = weights / np.sum(np.absolute(weights))
            self.pwc[catid] = weights.reshape((len(self.wvocab.w2i), ))

    def __count_words_by_category(self, train):
        """ カテゴリごとの単語の出現回数に変換を加えたスコアを返す。

        単語の出現回数に、以下の変換を施し返す。
            ・TF-Transform: 単語の出現頻度にlogをとり平滑化
            ・IDF-Trasform: IDF値を乗じることで頻出語の影響を緩和
            ・Lenght-Normalization: 文書の長さの影響を緩和するために文書長で正規化

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていない

        Returns:
            dict: カテゴリをkey, 単語のスコアを格納したcsr_matrixをvalueとして持つdict
        """

        cat2scores = {}
        for cat, docs in train.items():
            catid = self.cvocab.w2i[cat]
            data = np.asarray([], dtype=np.float32)
            row_idx = np.asarray([], dtype=np.int32)
            col_idx = np.asarray([], dtype=np.int32)
            for i, words in enumerate(docs):
                cnt = Counter(
                        [self.wvocab.w2i[w] for w in words
                            if w in self.wvocab.w2i]
                )
                # TF, and IDF-Transformation
                tfidf = np.asarray(
                        [np.log(c + 1) * self.idf[wid]
                            for wid, c in sorted(cnt.items())],
                        dtype=np.float32
                )
                # Length-Normalization
                norm_const = np.linalg.norm(tfidf)
                norm_tfidf = tfidf / norm_const
                data = np.concatenate((data, norm_tfidf))
                row_idx = np.concatenate(
                        (row_idx, [i for _ in range(len(cnt))])
                )
                col_idx = np.concatenate((col_idx, sorted(cnt.keys())))
            cat2scores[catid] = sp.csr_matrix(
                    (data, (row_idx, col_idx)),
                    shape=(len(docs), len(self.wvocab.w2i))
            )

        return cat2scores

    def __count_words_by_complement_category(self, target_catid, cat2scores):
        """ target_catで与えられたカテゴリ以外の単語のスコアの合計を求める。

        Args:
            target_cat (int): ターゲットカテゴリ
            cat2scores (dict): カテゴリをkey, 単語のスコアを格納したcsr_matrixをvalueとして持つdict

        Returns:
            csr_matrix: ターゲットカテゴリ以外の単語のスコアの合計を格納したカウンタ
        """

        ccat_scores = sp.csr_matrix(
                (1, len(self.wvocab.w2i)),
                dtype=np.float32
        )
        for catid, scores in cat2scores.items():
            if catid != target_catid:
                ccat_scores += sum(scores)
        return ccat_scores

    def __make_idf(self, train):
        """ IDFを計算する。

        Args:
            train (dict): カテゴリをkey, 各カテゴリの訓練データをvalueとして持つdict
            　　　　　　　訓練データは、各文書の単語リストを要素として持つリスト
                        　単語リストはuniqされていることが前提

        """

        global_uniq_cnt = Counter()
        for cat, docs in tqdm(train.items()):
            for words in docs:
                uniq_cnt = Counter(list(set(words)))
                global_uniq_cnt += uniq_cnt

        N = sum([len(docs) for docs in train.values()])
        self.idf = {
                self.wvocab.w2i[w]: np.log(N / c)
                for w, c in global_uniq_cnt.items()
                if w in self.wvocab.w2i
        }

    def predict(self, test, get_score=False):
        """ 学習したパラメータを用いて、テキストのカテゴリを分類する。

        Args:
            test (list): 予測を行う各文書の単語リストを要素として持つリスト
            get_score (bool): 予測ラベルのスコアを返すか否か

        Returns:
            list: 予測結果のラベル系列
            float: 予測ラベルのスコア(get_scoreがTrueの時のみ)
        """

        SCORE_IDX = 1
        catid_scores = []
        for words in tqdm(test):
            wordids = [
                    self.wvocab.w2i[w] for w in words if w in self.wvocab.w2i
            ]
            cnt = Counter(wordids)
            sorted_cnt = sorted(cnt.items())
            idx_arr = [idx for (idx, _) in sorted_cnt]
            cnt_arr = [c for (_, c) in sorted_cnt]
            wordids = np.zeros((len(self.wvocab.w2i),), dtype=np.float32)
            wordids[idx_arr] = cnt_arr
            _catid_scores = [
                    (catid, wordids.dot(weights))
                    for catid, weights in self.pwc.items()
            ]
            catid_scores.append(
                    min(_catid_scores, key=lambda x: x[SCORE_IDX])
            )

        if get_score:
            return [(self.cvocab.i2w[catid], score)
                    for (catid, score) in catid_scores]
        else:
            return [self.cvocab.i2w[catid] for (catid, score) in catid_scores]
