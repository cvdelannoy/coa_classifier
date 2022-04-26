from pathlib import Path
from matplotlib import pyplot as plt
from db_building.AbfData import AbfData

# Plot duration of events
coa_folder = Path('/home/noord087/lustre_link/capita_selecta/data/120_mv_trans/test')
coa_files = list(coa_folder.iterdir())
event_dict = {}
fig, ax = plt.subplots()
for file in coa_files:
    if file.name[:4].lower() in event_dict:
        print('Skipping')
        continue
    print(f'Processing {file}')
    abf = AbfData(file, lowpass_freq=100)
    if abf.coa_type not in event_dict:
        event_dict[abf.coa_type] = [[], []]
        event_lengths, rel_blockades = abf.get_fingerprints()
        event_dict[abf.coa_type][0].extend(event_lengths)
        event_dict[abf.coa_type][1].extend(rel_blockades)
    if len(event_dict) == 4:
        break

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
