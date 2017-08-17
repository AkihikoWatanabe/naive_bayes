# coding=utf-8

"""
語彙を扱う機能を提供するモジュール
"""

from collections import defaultdict


class Vocab():
    """ 語彙情報を扱うクラス

    Attributes:
        size (int): 語彙のサイズ
        w2i (defaultdict): 単語と単語インデクスのマップ
        i2w (list): 単語インデクスと単語のマップ
        full (bool): 語彙が上限まで登録されているか否か
    """

    def __init__(self, size):
        """ 語彙サイズを受け取り、マップの初期化を行う。

        Args:
            size(int): 語彙のサイズ
        """

        self.size = size
        self.w2i = defaultdict(lambda: len(self.w2i))
        self.i2w = [''] * self.size
        self.full = False

    def is_full(self):
        """ Vocabに登録されている単語が、語彙サイズの上限に達しているか否か判定し返す。

        Returns:
            bool: 上限に達していたらTrue, そうでなければFalse
        """

        return self.full

    def add(self, word):
        """ 単語を語彙に追加する

        Args:
            word (str): 語彙に追加する単語
        """

        if not self.full:
            idx = self.w2i[word]
            self.i2w[idx] = word
            self.full = True if len(self.w2i) == self.size else False

    def set_w2i(self, w2i):
        """ 単語と単語インデクスのマップを外部から読み込む

        Args:
            w2i (dict): 単語と単語インデクスのマップ
        """

        self.w2i = w2i

    def set_i2w(self, i2w):
        """ 単語のインデクスと単語のマップを外部から読み込む

        Args:
            i2w (dict): 単語インデクスと単語のマップ
        """

        self.i2w = i2w
