import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyabf
import pyabf.filter


def coa_to_one_hot(coa_type):
    """Returns one hot encoding based on COA type.
    :parameter coa_type: strings such as 'coa4' or 'coa6'
    """
    valid_coas = [3, 4, 5, 6]
    coa_number = int(coa_type[3])
    one_hot = np.zeros_like(valid_coas)
    one_hot[valid_coas.index(coa_number)] = 1
    return one_hot


class AbfData:
    """Class for squiggle data extracted from an axon binary file
    :param abf_fn: abf file name
    :param normalization: If true, perform normalization
    :type normalization: bool
    :param lowpass_freq: Frequency to use for low pass filter in KHz
    :type lowpass_freq: float
    :param baseline_fraction: Fraction of baseline that current needs to reach
                              to be labeled as positive event
    :type baseline_fraction: float
    """
    def __init__(self, abf_fn, normalization=False, lowpass_freq=80,
                 baseline_fraction=0.65):
        self.abf_fn = Path(abf_fn)
        self.normalization = normalization
        # Convert from KHz to ms
        self.smoothing_sigma = 1 / lowpass_freq
        self.raw = None
        self.baseline_level = np.median(self.raw)
        self.pos_events = self.set_pos_events(baseline_fraction)
        # coa_type describes if a coa 3,4,5,6. Currently extracted from filename
        self.coa_type = self.abf_fn.name[:4].lower()


    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, _):
        """The raw signal, after applying low-pass filter"""
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

    def set_pos_events(self, fraction):
        """Find all events where current drops below theshold,
        and set them as positive events. Output is saved in self.pos_events.

        :param fraction: Fraction of boseline to use as cutoff for positive event
        :return: List of all indices in raw signal that contain positive events.
        """
        cutoff = fraction * self.baseline_level
        event_ids = np.where(self.raw < cutoff)[0]
        # Ugly boiii
        self.flat_pos_indices = event_ids
        step_list = np.diff(event_ids)
        cut_points = np.where(step_list > 1)[0]
        cut_points = cut_points + 1
        events = np.split(event_ids, cut_points)
        # Keep events only of certain length
        return [event for event in events if len(event) > 13 and len(event) < 5e4]

    def get_event_lengths(self):
        """Get duration of events"""
        event_lengths = []
        for event in self.pos_events:
            start_idx = event[0]
            end_idx = event[-1]
            event_length = end_idx - start_idx
            # Convert to seconds
            event_length *= 2e-6
            event_lengths.append(event_length)

        return event_lengths

    def get_event_relative_blockades(self):
        rel_blockades = []
        for event_ix in self.pos_events:
            start_ix = event_ix[0]
            end_ix = event_ix[-1]
            event = self.unfiltered_raw[start_ix:end_ix]
            block_amplitude = self.baseline_level - event[len(event) // 2]
            # block_amplitude = self.baseline_level - np.median(event)
            relative_block = min(1, block_amplitude / self.baseline_level)
            rel_blockades.append(relative_block)
        return rel_blockades

    def get_fingerprints(self):
        """Return event length and relative blockade of all events"""
        event_lengths = self.get_event_lengths()
        rel_blockades = self.get_event_relative_blockades()
        assert len(event_lengths) == len(rel_blockades)
        return event_lengths, rel_blockades

    def plot_hist_pos(self):
        """Plot a histogram of event duration lengths"""
        event_lengths = self.get_event_lengths()
        plt.hist(event_lengths, bins=100)
        plt.show()

    def plot_full_signal(self):
        """Plot the complete raw signal"""
        plt.plot(self.raw)
        plt.ylabel('pA')
        plt.xlabel('Seconds')
        plt.show()

    def get_pos(self, unfiltered=False, take_one=False):
        """Get positive events

        :param unfiltered: If true, do not apply low-pass filter
        :param take_one: If true, return only one positive
        :return: list with positive events
        """
        pos_list = []
        if take_one:
            random.shuffle(self.pos_events)
        for event in self.pos_events:
            # Provide a small part of the baseline before and after the event
            start_idx = event[0] - 15
            end_idx = event[-1] + 15
            if take_one:
                return self.unfiltered_raw[start_idx: end_idx]
            if unfiltered:
                pos_list.append(self.unfiltered_raw[start_idx: end_idx])
            else:
                pos_list.append(self.raw[start_idx: end_idx])
        return pos_list

    def get_neg(self, width, nb_neg, unfiltered=False):
        """Get negative examples from signal, i.e. snippets of the baseline"""
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
