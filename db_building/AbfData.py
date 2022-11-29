import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pyabf
import pyabf.filter
from scipy.signal import butter, sosfilt
from resources.helper_functions import normalize_raw_signal
import os

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
                 baseline_fraction=0.65, event_type_dict={}):
        self.data_type = os.path.splitext(abf_fn)[1]
        self.min_duration, self.max_duration = 25e-6, 0.1  # duration thresholds in s
        self.abf_fn = Path(abf_fn)
        self.normalization = normalization
        # Convert from KHz to ms
        self.lowpass_freq = lowpass_freq
        self.smoothing_sigma = 1 / lowpass_freq
        self.raw = None
        self.baseline_level = np.median(self.raw)
        self.pos_events = self.set_pos_events(baseline_fraction)
        # coa_type describes if a coa 3,4,5,6. Currently extracted from filename
        self.coa_type = event_type_dict.get(self.abf_fn.name[:4].lower(), self.abf_fn.name[:4].lower())

    @property
    def raw(self):
        return self._raw

    @raw.setter
    def raw(self, _):
        """The raw signal, after applying low-pass filter"""
        if self.abf_fn.suffix == '.atf':
            abf = pyabf.ATF(self.abf_fn)
        elif self.abf_fn.suffix == '.npz':
            self.pos_events = list(np.load(self.abf_fn).values())
            self._raw = np.concatenate(self.pos_events)
            # self.pos_events = [normalize_raw_signal(x, self.normalization) for x in np.load(self.abf_fn).values()]
            return
        else:
            abf = pyabf.ABF(self.abf_fn)
        self.sample_rate = abf.sampleRate
        self.ds = round(self.sample_rate / (self.lowpass_freq * 10e2))
        self.post_decimation_step_size = self.ds / self.sample_rate
        self.unfiltered_raw = abf.data[0].copy()
        pyabf.filter.gaussian(abf, self.smoothing_sigma, channel=0)
        abf.setSweep(0)

        # Drop NaNs because they show up at the edges due to smoothing
        butter_filter = butter(100, 200 * 1000, 'lowpass', fs=self.sample_rate, output='sos')
        sig_filtered = sosfilt(butter_filter, self.unfiltered_raw)
        sig_filtered = abf.data[0].copy()

        nanbool = ~np.isnan(sig_filtered)
        self._raw = sig_filtered[nanbool]

        # Decimation
        # self.raw_decimated = self.decimate(abf.sweepY[nanbool], self.ds)
        self.raw_decimated = self.decimate(self._raw, self.ds)
        self.time_vector = abf.sweepX[nanbool]
        self.time_vector_decimated = self.decimate(abf.sweepX[nanbool], self.ds)

    @staticmethod
    def decimate(seq, ds):
        if (trunc_len := len(seq) % ds):
            seq_trunc = seq[:-trunc_len]
        else:
            seq_trunc = seq
        return np.median(seq_trunc.reshape((-1, ds)), -1)

    def set_pos_events(self, fraction):
        """Find all events where current drops below theshold,
        and set them as positive events. Output is saved in self.pos_events.

        :param fraction: Fraction of baseline to use as cutoff for positive event
        :return: List of all indices in raw signal that contain positive events.
        """
        if self.data_type == '.npz':
            return self.pos_events
        cutoff = fraction * self.baseline_level
        event_ids = np.where(self.raw_decimated < cutoff)[0]
        self.flat_pos_indices = event_ids
        step_list = np.diff(event_ids)
        cut_points = np.where(step_list > 1)[0]
        cut_points = cut_points + 1
        events = np.split(event_ids, cut_points)
        # Keep events only of certain length
        events_correct_length = [event for event in events
                if self.min_duration <
                self.time_vector_decimated[event[-1]] - self.time_vector_decimated[event[0]] + self.post_decimation_step_size
                < self.max_duration]

        event_durations = np.array([self.time_vector_decimated[event[-1]] - self.time_vector_decimated[
            event[0]] + self.post_decimation_step_size
                           for event in events_correct_length])

        return events_correct_length

    def get_event_lengths(self):
        """Get duration of events"""
        event_lengths = []
        for event in self.pos_events:
            start_idx = event[0]
            end_idx = event[-1]
            event_length = end_idx - start_idx
            # # Convert to seconds
            # event_length *= 2e-6
            event_lengths.append(event_length)
        return event_lengths

    def get_event_relative_blockades(self):
        rel_blockades = []
        for event_ix in self.pos_events:
            start_ix = event_ix[0]
            end_ix = event_ix[-1]
            event = self.unfiltered_raw[start_ix:end_ix]
            # block_amplitude = self.baseline_level - event[len(event) // 2]
            block_amplitude = self.baseline_level - np.median(event)
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
        if self.data_type == '.npz':
            return self.pos_events
        pos_list = []
        if take_one:
            random.shuffle(self.pos_events)
        for event in self.pos_events:
            # Provide a small part of the baseline before and after the event
            start_idx = event[0] - 15
            end_idx = event[-1] + 15
            if take_one:
                return normalize_raw_signal(self.unfiltered_raw[start_idx: end_idx], self.normalization)
            if unfiltered:
                pos_list.append(normalize_raw_signal(self.unfiltered_raw[start_idx: end_idx], self.normalization))
            else:
                pos_list.append(normalize_raw_signal(self.raw[start_idx: end_idx], self.normalization))
        return pos_list

    def get_baseline(self):
        out_list = []
        start_idx = 0
        for event in self.pos_events:
            end_idx = event[0] - 15
            out_list.append((self.unfiltered_raw[start_idx:end_idx]))
            start_idx = event[-1] + 15
        out_list.append(self.unfiltered_raw[start_idx:])
        return out_list

    def get_pos_blockade(self, unfiltered=False):
        if self.data_type == '.npz':
            return [np.mean(x) for x in self.pos_events]
        pos_list = []
        for event in self.pos_events:
            # Provide a small part of the baseline before and after the event
            start_idx = np.argmin(np.abs(self.time_vector - self.time_vector_decimated[event[0]])) - (self.ds // 2)
            end_idx = np.argmin(np.abs(self.time_vector - self.time_vector_decimated[event[-1]])) + (self.ds // 2)
            start_idx -= 15
            end_idx += 15
            if unfiltered:
                snippet = normalize_raw_signal(self.unfiltered_raw[start_idx: end_idx], self.normalization)
            else:
                snippet = normalize_raw_signal(self.raw[start_idx: end_idx], self.normalization)
            pos_list.append(np.mean(snippet))
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
