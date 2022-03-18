import AbfData
from pathlib import Path
import numpy as np
import random
from numpy.random import choice
import matplotlib.pyplot as plt

# Parts taken from https://github.com/swharden/pyABF/blob/master/docs/advanced/creating-waveforms/src/synth-atf.py

ATF_HEADER="""
ATF	1.0
8	2
"AcquisitionMode=Episodic Stimulation"
"Comment="
"YTop=2000"
"YBottom=-2000"
"SyncTimeUnits=20"
"SweepStartTimesMS=0.000"
"SignalsExported=IN 0"
"Signals="	"IN 0"
"Time (s)"	"Trace #1"
""".strip()

def create_atf(data, filename, rate=20000):
    """Save a waveform array as an ATF 1.0 file."""
    out = ATF_HEADER
    for i, val in enumerate(data):
        out += "\n%.05f\t%.05f" % (i/rate, val)
    with open(filename, 'w+') as f:
        f.write(out)
        print("wrote",filename)
    return

class InSilicoGenerator:
    def __init__(self, file_list):
        """Generate in silico squiggles of a ratio that you determine yourself
        :param file_list: List of abf files
        """
        self.coa4 = []
        self.coa5 = []
        self.coa6 = []
        self.load_abfs(file_list)
        self.squiggle_length = 60*50000  # 50 khz for 60 seconds
        self.squiggle = np.empty(self.squiggle_length)
        self.squiggle[:] = np.nan
        self.all_coas = [self.coa4, self.coa5, self.coa6]
        #  In the ground truth vector:
        # -4, -5, -6 are negatives for coa4, 5, 6
        #  4,  5,  6 are positives for coa4, 5, 6
        self.ground_truth_vector = np.empty_like(self.squiggle)

    def load_abfs(self, file_list):
        for file in file_list:
            if 'A4' in file.name:
                self.coa4.append(AbfData(file))
            elif 'A5' in file.name:
                self.coa5.append(AbfData(file))
            elif 'A6' in file.name:
                self.coa6.append(AbfData(file))

    def fill_squiggle(self, coa4_count, coa5_count, coa6_count, unfiltered=True,
                      insert_prob=0.939):
        """
        :param coa4_count: number of coa4 events
        :param coa5_count: number of coa5 events
        :param coa6_count: number of coa6 events
        """
        current_index = 0
        # Keep track of how many coa4,5,6 are inserted into the squiggle and what the target count is
        count_dict = {4: [0, coa4_count],
                      5: [0, coa5_count],
                      6: [0, coa6_count]}
        while current_index < len(self.squiggle):
            fragment_width = random.randint(100, 3000)
            coa_type = random.choices(self.all_coas,
                                      [coa4_count, coa5_count, coa6_count],
                                      k=1)[0]
            coa_object = random.choice(coa_type)
            fragment_id = int(coa_object.abf_fn.stem[-1])
            if count_dict[fragment_id][0] < count_dict[fragment_id][1]:
                positive_example = random.random() > insert_prob
            else:
                positive_example = False

            if current_index + fragment_width > self.squiggle_length:
                # Edge case: always fill final bit of squiggle with noise
                positive_example = False
                fragment_width = self.squiggle_length - current_index
            if positive_example:
                squiggle_fragment = coa_object.get_pos(fragment_width,
                                                       unfiltered=unfiltered,
                                                       take_one=True)
                count_dict[fragment_id][0] += 1
            else:
                squiggle_fragment = coa_object.get_neg(fragment_width,
                                                       nb_neg=1,
                                                       unfiltered=unfiltered)[0]
                fragment_id *= -1
            end_index = current_index + fragment_width
            self.squiggle[current_index:end_index] = squiggle_fragment
            self.ground_truth_vector[current_index:end_index] = fragment_id
            current_index += fragment_width
        print('cOA type: [number of events in in silico squiggle, target number]')
        print(count_dict)
        print('If it is too far off, change insert_prob')
        return self.squiggle, self.ground_truth_vector

def but_im_not_a_wrapper(files, coa4, coa5, coa6, n):
    generator = InSilicoGenerator(files)
    for i in range(n):
        simulated_squiggle, ground_truth = generator.fill_squiggle(coa4, coa5,
                                                                   coa6)
        try:
            create_atf(simulated_squiggle,
                       filename=Path(f'/mnt/c/Users/benno/PycharmProjects/baseless/data/{coa4}_{coa5}_{coa6}_sim_{i}.atf'),
                       rate=50000)
        except ValueError:
            continue


if __name__ == '__main__':
    files = [Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_cOA4.abf"),
             Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_2_cOA4.abf"),
             Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_cOA5.abf"),
             Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_2_cOA5.abf"),
             Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_cOA6.abf"),
             Path(r"/mnt/c/Users/benno/Downloads/zooi/+120mV_2_cOA6.abf")]
    combinations = ((30, 30, 30), (60, 15, 15), (15, 60, 15), (15, 15, 60))
    n = 10
    for combination in combinations:
        c4, c5, c6 = combination
        but_im_not_a_wrapper(files, c4, c5, c6, n)

    # plt.plot(simulated_squiggle)
    # plt.show()
