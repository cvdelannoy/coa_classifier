from pathlib import Path
from matplotlib import pyplot as plt
from db_building.AbfData import AbfData

# Plot duration of events
coa_folder = Path('/home/noord087/lustre_link/capita_selecta/data/120_mv_trans_no_coa5/train')
coa_files = list(coa_folder.iterdir())[:14]
event_dict = {}
fig, ax = plt.subplots()
for file in coa_files:
    print(f'Processing {file}')
    abf = AbfData(file)
    if abf.coa_type not in event_dict:
        event_dict[abf.coa_type] = [[], []]
    event_lengths, rel_blockades = abf.get_fingerprints()
    event_dict[abf.coa_type][0].extend(event_lengths)
    event_dict[abf.coa_type][1].extend(rel_blockades)
for abf_type, measurements in event_dict.items():
    ax.scatter(measurements[0], measurements[1], alpha=0.1, label=abf_type)
ax.set_xscale('log')
fig.legend()
plt.show()
