# coding=utf-8

"""
オブジェクトのSerializeに関する機能を提供するモジュール
"""

import pickle as cPickle
import gzip


class Serializer():
    """ オブジェクトをSerializeする機能を提供するクラス
    """

    @staticmethod
    def dump_data(frms, file_path, suffix=".pkl.gz"):
        """ dump data to file_path using cPickle

        Args:
            frms(list): dump data
            file_path(str): path to dump
        """

        with gzip.open(file_path+suffix, 'wb') as gf:
            [cPickle.dump(frm, gf, cPickle.HIGHEST_PROTOCOL) for frm in frms]

    @staticmethod
    def load_data(file_path, suffix=".pkl.gz"):
        """ load dump data from file_path

        Args:
            file_path(str): path for load
        """

        data = []
        with gzip.open(file_path+suffix, 'rb') as gf:
            while True:
                try:
                    data.append(cPickle.load(gf))
                except EOFError:
                    break
        return data[0] if len(data) == 1 else data
