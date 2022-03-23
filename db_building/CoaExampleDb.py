import ZODB, ZODB.FileStorage, BTrees.IOBTree
from os.path import isfile
import numpy as np
import random
from db_building.AbfData import AbfData
from copy import deepcopy


class ExampleDb(object):
    """
    A class for a database storing training examples for a neural network
    """
    def __init__(self, **kwargs):
        self._db = None
        self._db_empty = True
        self.nb_pos = 0
        if not isfile(kwargs['db_name']):
            self.width = kwargs['width']
        self.db_name = kwargs['db_name']

        self.read_only = kwargs.get('read_only', False)
        self.db = self.db_name

    def add_training_read(self, training_read):
        """Add training read with cOA events

        :param training_read: Training read
        :type training_read: AbfData
        """
        label = training_read.get_one_hot()

        # --- add positive examples ---
        with self._db.transaction() as conn:
            pos_examples = training_read.get_pos(self.width)
            for ex in pos_examples:
                insert_index = len(conn.root.pos)
                conn.root.pos[insert_index] = ex
                conn.root.labels[insert_index] = label
            self.nb_pos = conn.root.pos.maxKey() if conn.root.pos else 0

        nb_examples = len(pos_examples)
        print(f'Added {nb_examples} examples for {training_read.coa_type}')
        if self._db_empty:
            if nb_examples > 0:
                self._db_empty = False

        return nb_examples

    def get_training_set(self, size=None):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """
        # TODO is this actually balanced?
        if size is None or size > self.nb_pos:
            size = self.nb_pos
        ps = random.sample(range(self.nb_pos), size)

        with self._db.transaction() as conn:
            examples_pos = [(conn.root.pos[n], conn.root.labels[n]) for n in ps]

        random.shuffle(examples_pos)
        x_out, y_out = zip(*examples_pos)
        return x_out, np.array(y_out)

    def pack_db(self):
        self._db.pack()
        with self._db.transaction() as conn:
            self.nb_pos = conn.root.pos.maxKey() if conn.root.pos else 0

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
                self.nb_pos = len(conn.root.pos)
        else:
            with self._db.transaction() as conn:
                conn.root.width = self.width
                conn.root.pos = BTrees.IOBTree.BTree()
                conn.root.labels = BTrees.IOBTree.BTree()
        if self.nb_pos > 0:
            self._db_empty = False
