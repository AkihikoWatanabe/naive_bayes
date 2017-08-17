# coding=utf-8

NBParam = namedtuple(
        'NBParam',
        ('learning', 'ftr_slct', 'stopwords_num', 'vocab_size', 'alpha')
)


class NBParamWithDocstring(NBParam):
    """ ナイーブベイズ分類器で用いるパラメータを格納するnamedtuple

    Attributes:
        learning (str): 学習方法（MLE or MAP）
                        MLEは最尤推定、MAPはMAP推定を表す。
        stopwords_num (int): ストップワードの数
        vocab_size (int): 語彙のサイズ
        alpha (int): ディリクレ分布のハイパーパラメータ
                     単語の出現回数にどれだけ下駄を履かせるかに相当
    """

# 最尤推定を表す文字列
LEARNING_MLE = "MLE"

# MAP推定を表す文字列
LEARNING_MAP = "MAP"

# 特徴選択を行わないことを表す文字列
FS_NO = "NO_FS"

# 単語頻度による特徴選択を表す文字列
FS_FREQ = "FREQ"

# PMIによる特徴選択を表す文字列
FS_PMI = "PMI"

