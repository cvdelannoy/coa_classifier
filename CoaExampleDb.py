import ZODB, ZODB.FileStorage, BTrees.IOBTree
from os.path import isfile
import numpy as np
import random
from AbfData import AbfData
from copy import deepcopy


class ExampleDb(object):
    """
    Database storing training examples for a neural network
    :param db_name: path to database file
    :param width: width of fragments
    :param read_only (optional):
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
        """Add events of cOA to DB from training read

        :param training_read: Object containing a training read
        :type training_read: AbfData
        :return: Number of added events
        """
        label = training_read.get_one_hot()

        with self._db.transaction() as conn:
            # --- add positive examples (if any) ---
            pos_examples = training_read.get_pos(self.width)
            for _, ex in enumerate(pos_examples):
                insert_index = len(conn.root.pos)
                conn.root.pos[insert_index] = ex
                conn.root.labels[insert_index] = label
            nb_new_positives = len(pos_examples)

            # --- update record nb positive examples ---
            if self._db_empty:
                if nb_new_positives > 0:
                    self._db_empty = False
            if not self._db_empty:
                self.nb_pos = conn.root.pos.maxKey()
        return nb_new_positives

    def get_training_set(self, size=None):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """

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
            self.nb_pos = conn.root.pos.maxKey()

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
