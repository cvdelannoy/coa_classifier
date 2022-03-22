import numpy as np
import os
import re
import shutil

from glob import glob
from math import nan, log
from statistics import median

from bokeh.models import ColumnDataSource, LinearColorMapper, LabelSet, Range1d
from bokeh.plotting import figure

from math import pi
wur_colors = ['#E5F1E4', '#3F9C35']


def parse_input_path(in_dir, pattern=None, regex=None):
    if type(in_dir) != list: in_dir = [in_dir]
    out_list = []
    for ind in in_dir:
        if not os.path.exists(ind):
            raise ValueError(f'{ind} does not exist')
        if os.path.isdir(ind):
            ind = os.path.abspath(ind)
            if pattern is not None: out_list.extend(glob(f'{ind}/**/{pattern}', recursive=True))
            else: out_list.extend(glob(f'{ind}/**/*', recursive=True))
        else:
            if pattern is None: out_list.append(ind)
            elif pattern.strip('*') in ind: out_list.append(ind)
    if regex is not None:
        out_list = [fn for fn in out_list if re.search(regex, fn)]
    return out_list


def parse_output_path(location, clean=False):
    """
    Take given path name. Add '/' if path. Check if exists, if not, make dir and subdirs.
    """
    if location[-1] != '/':
        location += '/'
    if clean:
        shutil.rmtree(location, ignore_errors=True)
    if not os.path.isdir(location):
        os.makedirs(location)
    return location


def plot_timeseries(raw, base_labels, posterior, y_hat, start=0, nb_classes=2, reflines=(0.90, 0.95, 0.99)):

    nb_points = y_hat.size
    scaling = np.abs(raw.max() - raw.min()) / np.abs(posterior.max() - posterior.min())
    posterior = posterior * scaling
    translation = - posterior.min() + raw.min()
    posterior = posterior + translation

    reflines = np.array([log(rl / (1-rl)) for rl in reflines])
    reflines = reflines * scaling + translation
    source_dict = dict(
        raw=raw[:nb_points],
        posterior=posterior,
        event=list(range(len(y_hat))),
        cat=y_hat,
        cat_height=np.repeat(np.mean(raw), len(y_hat))
    )
    for ri, r in enumerate(reflines):
        source_dict[f'r{ri}'] = np.repeat(r, len(posterior))

    # Main data source
    source = ColumnDataSource(source_dict)

    # Base labels stuff
    base_labels_condensed = [base_labels[0]]
    bl_xcoords = []
    event_ends = []
    new_range = []
    for idx, bl in enumerate(base_labels):
        if bl == base_labels_condensed[-1]:
            new_range.append(idx)
        else:
            base_labels_condensed.append(bl)
            bl_xcoords.append(median(new_range))
            event_ends.append(idx)
            new_range = [idx]
    bl_xcoords.append(median(new_range))
    event_ends.append(new_range[-1])

    bl_source = ColumnDataSource(dict(
        base_labels=base_labels_condensed,
        event_ends= event_ends,
        x=bl_xcoords,
        y= np.repeat(raw.max(), len(bl_xcoords))
    ))

    base_labels_labelset = LabelSet(x='x', y='y', text='base_labels',
                                    source=bl_source,
                                    text_baseline='middle',
                                    angle=0.25*pi)

    # Plotting
    ts_plot = figure(title='Classified time series')
    ts_plot.grid.grid_line_alpha = 0.3
    ts_plot.xaxis.axis_label = 'nb events'
    ts_plot.yaxis.axis_label = 'current signal'
    y_range = raw.max() - raw.min()

    colors=wur_colors
    col_mapper = LinearColorMapper(palette=colors, low=0, high=nb_classes-1)
    ts_plot.rect(x='event', y='cat_height', width=1.05, height=y_range, source=source,
                 fill_color={
                     'field': 'cat',
                     'transform': col_mapper
                 },
                 line_color=None)

    # Event ending lines
    ts_plot.rect(x='event_ends',
                 y=raw.mean(),
                 height=y_range, width=0.01, line_color='black', source=bl_source)


    ts_plot.add_layout(base_labels_labelset)
    ts_plot.line(x='event', y='raw', source=source)
    ts_plot.line(x='event', y='posterior', color='red', source=source)
    for i in range(len(reflines)):
        ts_plot.line(x='event', y=f'r{i}', color='grey', source=source)
    ts_plot.plot_width = 1500
    ts_plot.plot_height = 500
    ts_plot.x_range = Range1d(start, start+100)
    return ts_plot
