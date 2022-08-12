import random
from os.path import isfile

import ZODB
import ZODB.FileStorage
import numpy as np
from persistent.mapping import PersistentMapping

from db_building.AbfData import AbfData


class ExampleDb(object):
    """
    Database storing training examples for a neural network
    :param db_name: path to database file
    :param read_only (optional):
    """
    def __init__(self, **kwargs):
        self._db = None
        self._db_empty = True
        self.nb_pos = 0
        self.db_name = kwargs['db_name']

        self.read_only = kwargs.get('read_only', False)
        self.db = self.db_name
        self.event_type_dict = kwargs.get('event_type_dict', {})
        self.target_to_index = {x: i for i, x in enumerate(np.unique(list(self.event_type_dict.values())))}
        self.nb_targets = len(self.target_to_index)

    def coa_to_one_hot(self, coa_type):
        """Returns one hot encoding based on COA type.
        :parameter coa_type: strings such as 'coa4' or 'coa6'
        """
        one_hot = np.zeros(self.nb_targets, dtype=int)
        one_hot[self.target_to_index[coa_type]] = 1
        return one_hot

    def add_training_read(self, training_read, unfiltered):
        """Add training read with cOA events

        :param training_read: Object containing a training read
        :type training_read: AbfData
        :param unfiltered: If true, return squiggle that has not been passed
                           through low-pass filter
        :return: Number of added events
        """
        label = training_read.coa_type

        with self._db.transaction() as conn:
            if label not in conn.root.examples:
                conn.root.examples[label] = []

            # --- add positive examples (if any) ---
            pos_examples = training_read.get_pos(unfiltered=unfiltered)
            for ex in pos_examples:
                conn.root.examples[label].append(ex)
            nb_new_positives = len(pos_examples)

            # --- update record nb positive examples ---
            if self._db_empty and nb_new_positives > 0:
                self._db_empty = False
                self.nb_pos = nb_new_positives
            else:
                self.nb_pos += nb_new_positives
        # print(f'Added {nb_new_positives} examples for {training_read.coa_type}')

        return nb_new_positives

    def get_training_set(self, size=None, oversampling=False):
        """
        Return a balanced subset of reads from the DB
        :param size: number of reads to return
        :return: lists of numpy arrays for training data(x_out) and labels (y_out)
        """
        if size is None or size > self.nb_pos:
            size = self.nb_pos

        training_set_x = []
        training_set_y = []

        with self._db.transaction() as conn:
            total_coas = len(conn.root.examples.keys())
            # Get which class has least examples.
            # Get how many items are expected per key based on set size
            items_per_key = int(size / total_coas)
            if oversampling:
                nr_of_examples = items_per_key
            else:
                least_examples_in_class = min([len(v) for v in conn.root.examples.values()])
                nr_of_examples = min(items_per_key, least_examples_in_class)
            print(f'Using {nr_of_examples} examples per class')
            for coa, signals in conn.root.examples.items():
                one_hot = self.coa_to_one_hot(coa)
                signals_selected = list(np.random.choice(np.array(signals, dtype=object), size=nr_of_examples,
                                                        replace=nr_of_examples > len(signals)))
                training_set_x.extend(signals_selected)
                training_set_y.extend([one_hot] * nr_of_examples)

        # Shuffle
        c = list(zip(training_set_x, training_set_y))
        random.shuffle(c)
        training_set_x, training_set_y = zip(*c)

        return training_set_x, training_set_y

    def pack_db(self):
        self._db.pack()
        with self._db.transaction() as conn:
            self.nb_pos = sum([len(i) for i in conn.root.examples.values()])

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
        storage = ZODB.FileStorage.FileStorage(db_name,
                                               read_only=self.read_only,
                                               pack_keep_old=False)
        self._db = ZODB.DB(storage)
        if is_existing_db:
            with self._db.transaction() as conn:
                # Get total number of entries in database
                self.nb_pos = sum([len(i) for i in conn.root.examples.values()])
        else:
            with self._db.transaction() as conn:
                conn.root.examples = PersistentMapping()
        if self.nb_pos > 0:
            self._db_empty = False
