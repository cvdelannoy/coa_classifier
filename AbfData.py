from pathlib import Path
import pyabf
import pyabf.filter
import numpy as np
import matplotlib
# Workaround to view interactive figures
matplotlib.use('WebAgg')
import matplotlib.pyplot as plt
from pathlib import Path
import random


class AbfData:
    """Class for squiggle data extracted from an axon binary file
    :param abf_fn: abf file name
    :param normalization: If true, perform normalization
    :type normalization: bool
    :param lowpass_freq: Frequency to use for low pass filter in KHz
    :type lowpass_freq: float
    """
    def __init__(self, abf_fn, normalization=False, lowpass_freq=80,
                 baseline_threshold=0.65):
        self.abf_fn = Path(abf_fn)
        self.normalization = normalization
        # Convert from KHz to ms
        self.smoothing_sigma = 1 / lowpass_freq
        self.raw = None
        self.baseline_level = np.median(self.raw)
        self.cutoff = self.baseline_level * baseline_threshold
        self.pos_events = self.set_pos_events()

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, _):
        if self.abf_fn.suffix == '.atf':
            abf = pyabf.ATF(self.abf_fn)
        else:
            abf = pyabf.ABF(self.abf_fn)
        self.unfiltered_raw = abf.sweepY
        pyabf.filter.gaussian(abf, self.smoothing_sigma)
        abf.setSweep(0)
        # Drop NaNs because they show up at the edges due to smoothing
        self._raw = abf.sweepY[~np.isnan(abf.sweepY)]
        self.unfiltered_raw = self.unfiltered_raw[~np.isnan(abf.sweepY)]
        self.time_vector = abf.sweepX[~np.isnan(abf.sweepY)]

    def set_pos_events(self):
        event_ids = np.where(self.raw < self.cutoff)[0]
        # Ugly boiii
        self.flat_pos_indices = event_ids
        step_list = np.diff(event_ids)
        cut_points = np.where(step_list>1)[0]
        cut_points = cut_points + 1
        events = np.split(event_ids, cut_points)
        return [event for event in events if len(event) > 2]

    def plot_hist_pos(self, width):
        event_lengths = []
        for event in self.pos_events:
            start_idx = event[0]
            end_idx = event[-1]
            event_length = end_idx - start_idx
            event_lengths.append(event_length)
            room_left = width - event_length
        plt.hist(event_lengths, bins=100, range=(0, 500))
        plt.show()

    def get_pos(self, width: int, unfiltered=False, take_one=False):
        """Get a list of all cOA passing events in this trace

        :param width: Width of the events to output
        :param unfiltered: If true, do not apply low-pass filter
        :param take_one: If true, return one randomly selected positive event
        :return: list of all positive events
        """
        pos_list = []
        if take_one:
            random.shuffle(self.pos_events)
        for event in self.pos_events:
            start_idx = event[0]
            end_idx = event[-1]
            event_length = end_idx - start_idx
            room_left = width - event_length
            if room_left > 0:
                random_offset = random.randint(0, room_left)
                start_idx -= random_offset
                end_idx += room_left - random_offset
                assert (end_idx - start_idx) == width
                if take_one:
                    return self.unfiltered_raw[start_idx: end_idx]
                if unfiltered:
                    pos_list.append(self.unfiltered_raw[start_idx: end_idx])
                else:
                    pos_list.append(self.raw[start_idx: end_idx])
        return pos_list

    def get_neg(self, width, nb_neg, unfiltered=False):
        neg_list = []
        for i in range(nb_neg * 100): # take 100 attempts for each neg
            random_idx = random.randint(0, len(self.raw)-width)
            candidate_indices = np.arange(random_idx, random_idx+width)
            if not np.any(np.isin(candidate_indices, self.flat_pos_indices)):
                if unfiltered:
                    neg_list.append(self.unfiltered_raw[candidate_indices])
                else:
                    neg_list.append(self.raw[candidate_indices])
            if len(neg_list) >= nb_neg:
                return neg_list


if __name__ == '__main__':
    # General inspection of what squiggle looks like
    local_file_path = Path(r"/home/bnoordijk/coa_data_mount/data/cOA6/cis/+120mV.abf")
    # local_file_path = Path(
    #     r"/home/bnoordijk/coa_data_mount/data/cOA3/cis/cOA3_cis_+120mV_1.abf")
    squiggle = AbfData(local_file_path, normalization=False, lowpass_freq=80)
    # print(squiggle.cutoff)
    # plt.plot(squiggle.raw)

    a = squiggle.get_pos(250, take_one=True)
    plt.plot(a)

    # # View squiggle durations
    # squiggle.plot_hist_pos(250)
    # a = squiggle.get_pos(2000, take_one=True)
    # b = squiggle.get_neg(2000, 1)
    # fig, axs = plt.subplots(2, 1)
    # axs[0].plot(a)
    # axs[1].plot(b)
    # plt.show()
    # # event_indices = squiggle.get_events(0.65)
    # # Plot events here for debugging
    # plt.subplot()
    # plt.plot(squiggle.time_vector, squiggle.raw)
    # plt.plot(squiggle.time_vector[event_indices], squiggle.raw[event_indices], '+')

    # Add correct labels and show plot
    plt.ylabel('pA')
    plt.xlabel('Seconds')
    plt.show()

