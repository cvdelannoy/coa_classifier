from pathlib import Path

from matplotlib import pyplot as plt

from db_building.AbfData import AbfData


def extract_fingerprints_from_files(files):
    """ From list of files, extract the relative blockade and event length

    :param files: Iterable of ABF files
    :return: dictionary with key: abf type and value: list of two
             characteristics.  The first item in the list is the event length,
             the second is the relative blockade.
    """
    event_dict = {}
    for file in files:
        if file.name[:4].lower() in event_dict:
            print('Skipping')
            continue
        print(f'Processing {file}')
        abf = AbfData(file, lowpass_freq=80)
        if abf.coa_type not in event_dict:
            event_dict[abf.coa_type] = [[], []]
            event_lengths, rel_blockades = abf.get_fingerprints()
            event_dict[abf.coa_type][0].extend(event_lengths)
            event_dict[abf.coa_type][1].extend(rel_blockades)
        if len(event_dict) == 4:
            break
    return event_dict

def plot_scatter_fingerprints(event_dict):
    """Plot scatterplot of event length and relative blockade per coa type

    :param event_dict: Dictionary with per-coa event fingerprints. Output
                       by extract_fingerprints_from_files
    """
    fig, ax = plt.subplots()
    for abf_type, measurements in event_dict.items():
        ax.scatter(measurements[0], measurements[1], alpha=0.5, label=abf_type,
                   marker='.')

    handles, labels = ax.get_legend_handles_labels()
    # sort both labels and handles by labels
    labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
    ax.legend(handles, labels)
    ax.set_xscale('log')
    ax.set_xlabel('Dwell time (s)')
    ax.set_ylabel('Relative blockade')
    # fig.legend()
    plt.tight_layout()
    plt.savefig('/home/noord087/lustre_link/capita_selecta/outputs/figures/sanity_check/fingerprints_amplitude_from_middle_lpf_100.png', dpi=300)
    # plt.show()


def main():
    # Plot duration of events
    coa_folder = Path('/home/noord087/lustre_link/capita_selecta/data/120_mv_trans/train')
    coa_files = list(coa_folder.iterdir())
    event_dict = extract_fingerprints_from_files(coa_files)
    plot_scatter_fingerprints(event_dict)

if __name__ == '__main__':
    main()
