import ZODB, ZODB.FileStorage, BTrees.IOBTree
from os.path import isfile
import numpy as np
import random
from copy import deepcopy


class ExampleDb(object):
    """
    A class for a database storing training examples for a neural network
    """
    def __init__(self, **kwargs):

        self._db = None
        self.neg_kmers = dict()  # Dict of lists, indices per encountered negative example k-mer
        self._db_empty = True
        self.nb_pos = 0
        self.nb_neg = 0
        if not isfile(kwargs['db_name']):
            self.target = kwargs['target']
            self.width = kwargs['width']
        self.db_name = kwargs['db_name']

        self.read_only = kwargs.get('read_only', False)
        self.db = self.db_name

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = target

    def add_training_read(self, training_read, non_target=None):
        """Add training read with positive and negative read examples of the
        target k-mer

        :param training_read: Object containing a training read
        :param uncenter_kmer: If set to true, will extract reads where the k-mer
                              is randomly places somewhere in the read. Not
                              always in the center
        :type uncenter_kmer: bool
        """
        with self._db.transaction() as conn:
            pos_examples = training_read.get_pos(self.width)
            for i, ex in enumerate(pos_examples):
                if non_target:
                    conn.root.neg[len(conn.root.neg)] = ex
                else:
                    conn.root.pos[len(conn.root.pos)] = ex
            self.nb_pos = conn.root.pos.maxKey() if conn.root.pos else 0
            self.nb_neg = conn.root.neg.maxKey() if conn.root.neg else 0

        # --- add positive examples (if any) ---
        nb_examples = len(pos_examples)

        # --- update record nb positive examples ---
        if self._db_empty:
            if nb_examples > 0:
                self._db_empty = False

        return nb_examples

    def add_non_target(self, examples, non_target_name, conn):
        for i, ex in enumerate(examples):
            if non_target_name in self.neg_kmers:  # CL: kept the name 'kmer', even though we're not detecting k-mers anymore
                self.neg_kmers[non_target_name].append(self.nb_neg + i)
            else:
                self.neg_kmers[non_target_name] = [self.nb_neg + i]
            conn.root.neg[len(conn.root.neg)] = ex
        return len(examples)

    def get_training_set(self, size=None):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """
        if size is None or size > self.nb_pos or size > self.nb_neg:
            size = min(self.nb_pos, self.nb_neg)
        nb_pos = size // 2
        nb_neg = size - nb_pos
        ps = random.sample(range(self.nb_pos), nb_pos)
        ns = random.sample(range(self.nb_neg), nb_neg)

        with self._db.transaction() as conn:
            examples_pos = [(conn.root.pos[n], 1) for n in ps]
            examples_neg = [(conn.root.neg[n], 0) for n in ns]
        data_out = examples_pos + examples_neg
        random.shuffle(data_out)
        x_out, y_out = zip(*data_out)
        return x_out, np.array(y_out)

    def pack_db(self):
        self._db.pack()
        with self._db.transaction() as conn:
            self.nb_pos = conn.root.pos.maxKey() if conn.root.pos else 0
            self.nb_neg = conn.root.neg.maxKey() if conn.root.neg else 0

    @property
    def db(self):
        return self._db

    @db.setter
    def db(self, db_name):
        """
        Construct ZODB database if not existing, store DB object
        :param db_name: name of new db, including path
        """
        is_existing_db = isfile(db_name)
        storage = ZODB.FileStorage.FileStorage(db_name, read_only=self.read_only)
        self._db = ZODB.DB(storage)
        if is_existing_db:
            with self._db.transaction() as conn:
                self.width = conn.root.width
                self.target = conn.root.target
                self.nb_pos = len(conn.root.pos)
                self.nb_neg = len(conn.root.neg)
        else:
            with self._db.transaction() as conn:
                conn.root.target = self.target[0]
                conn.root.width = self.width
                conn.root.pos = BTrees.IOBTree.BTree()
                conn.root.neg = BTrees.IOBTree.BTree()
        if self.nb_pos > 0:
            self._db_empty = False
