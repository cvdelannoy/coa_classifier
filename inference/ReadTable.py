import numpy as np
from multiprocessing import Process, Queue

from db_building.AbfData import AbfData
from inference.ReadManager import ReadManager


class ReadTable(object):

    def __init__(self, reads_dir, pos_reads_dir):
        """Table that keeps track of all reads in directory and what
        kmers they contain

        :param reads_dir: Directory that contains fast5 reads on which to run inference
        :param table_fn: File name of this table database
        :param pos_reads_dir: Directory where reads tested positive for sequence are stored
        :param kmers: List of kmers for which to search
        :param input_length: Length of signal that should be passed to model as input
        :param batch_size: Batch size for the model
        """
        self.pos_reads_dir = pos_reads_dir
        self.reads_dir = reads_dir
        self._pred_queue = Queue()
        self._new_read_queue = Queue()

    def init_table(self):
        manager_process = Process(target=ReadManager, args=(self._new_read_queue,
                                                            self._pred_queue,
                                                            self.reads_dir,
                                                            self.pos_reads_dir), name='read_manager')
        manager_process.start()
        return manager_process

    def get_read_to_predict(self, batch_size=4):
        """Get read and a kmer for which to scan the read file

        :return: Tuple of: (path to read fast5 file,
                 Read object for inference (split to desired input length),
                 k-mer as string for which to scna)
        """
        # try:
        #     read_fn = self._new_read_queue.get_nowait()  # stalls if queue is empty
        # except Empty:
        #     return None, None
        ii, fn_list = 0, []
        while ii < batch_size and not self._new_read_queue.empty():
            fn_list.append(self._new_read_queue.get_nowait())
            ii += 1
        if not len(fn_list): return None, None
        raw_list = []
        for read_fn in fn_list:
            raw = AbfData(read_fn).raw
            lr = len(raw)
            if lr > 1E6:  # todo: hack to avoid memory issues for squiggles of length up to millions
                raw_list.extend(np.array_split(raw, lr / 1E6))
            else:
                raw_list.append(raw)
        return fn_list, raw_list

    def update_prediction(self, fn, pred):
        for f, p in zip(fn, pred):
            self._pred_queue.put((f, p))
