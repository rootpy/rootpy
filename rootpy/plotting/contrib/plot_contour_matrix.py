# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import os
import tempfile
import shutil

from . import log; log = log[__name__]
from .. import Hist2D
from .gif import GIF

__all__ = [
    'plot_contour_matrix',
]

LINES = ['dashed', 'solid', 'dashdot', 'dotted']


def plot_contour_matrix(arrays,
                        fields,
                        filename,
                        weights=None,
                        sample_names=None,
                        sample_lines=None,
                        sample_colors=None,
                        color_map=None,
                        num_bins=20,
                        num_contours=3,
                        cell_width=2,
                        cell_height=2,
                        cell_margin_x=0.05,
                        cell_margin_y=0.05,
                        dpi=100,
                        padding=0,
                        animate_field=None,
                        animate_steps=10,
                        animate_delay=20,
                        animate_loop=0):
    """
    Create a matrix of contour plots showing all possible 2D projections of a
    multivariate dataset. You may optionally animate the contours as a cut on
    one of the fields is increased. ImageMagick must be installed to produce
    animations.

    Parameters
    ----------

    arrays : list of arrays of shape [n_samples, n_fields]
        A list of 2D NumPy arrays for each sample. All arrays must have the
        same number of columns.

    fields : list of strings
        A list of the field names.

    filename : string
        The output filename. If animatation is enabled
        ``animate_field is not None`` then ``filename`` must have the .gif
        extension.

    weights : list of arrays, optional (default=None)
        List of 1D NumPy arrays of sample weights corresponding to the arrays
        in ``arrays``.

    sample_names : list of strings, optional (default=None)
        A list of the sample names for the legend. If None, then no legend will
        be shown.

    sample_lines : list of strings, optional (default=None)
        A list of matplotlib line styles for each sample. If None then line
        styles will cycle through 'dashed', 'solid', 'dashdot', and 'dotted'.
        Elements of this list may also be a list of line styles which will be
        cycled through for the contour lines of the corresponding sample.

    sample_colors : list of matplotlib colors, optional (default=None)
        The color of the contours for each sample. If None, then colors will be
        selected according to regular intervals along the ``color_map``.

    color_map : a matplotlib color map, optional (default=None)
        If ``sample_colors is None`` then select colors according to regular
        intervals along this matplotlib color map. If ``color_map`` is None,
        then the spectral color map is used.

    num_bins : int, optional (default=20)
        The number of bins along both axes of the 2D histograms.

    num_contours : int, optional (default=3)
        The number of contour line to show for each sample.

    cell_width : float, optional (default=2)
        The width, in inches, of each subplot in the matrix.

    cell_height : float, optional (default=2)
        The height, in inches, of each subplot in the matrix.

    cell_margin_x : float, optional (default=0.05)
        The horizontal margin between adjacent subplots, as a fraction
        of the subplot size.

    cell_margin_y : float, optional (default=0.05)
        The vertical margin between adjacent subplots, as a fraction
        of the subplot size.

    dpi : int, optional (default=100)
        The number of pixels per inch.

    padding : float, optional (default=0)
        The padding, as a fraction of the range of the value along each axes to
        guarantee around each sample's contour plot.

    animate_field : string, optional (default=None)
        The field to animate a cut along. By default no animation is produced.
        If ``animate_field is not None`` then ``filename`` must end in the .gif
        extension and an animated GIF is produced.

    animate_steps : int, optional (default=10)
        The number of frames in the animation, corresponding to the number of
        regularly spaced cut values to show along the range of the
        ``animate_field``.

    animate_delay : int, optional (default=20)
        The duration that each frame is shown in the animation as a multiple of
        1 / 100 of a second.

    animate_loop : int, optional (default=0)
        The number of times to loop the animation. If zero, then loop forever.

    Notes
    -----

    NumPy and matplotlib are required

    """
    import numpy as np
    from .. import root2matplotlib as r2m
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    from matplotlib import cm
    from matplotlib.lines import Line2D

    # we must have at least two fields (columns)
    num_fields = len(fields)
    if num_fields < 2:
        raise ValueError(
            "record arrays must have at least two fields")
    # check that all arrays have the same number of columns
    for array in arrays:
        if array.shape[1] != num_fields:
            raise ValueError(
                "number of array columns does not match number of fields")

    if sample_colors is None:
        if color_map is None:
            color_map = cm.spectral
        steps = np.linspace(0, 1, len(arrays) + 2)[1:-1]
        sample_colors = [color_map(s) for s in steps]

    # determine range of each field
    low = np.vstack([a.min(axis=0) for a in arrays]).min(axis=0)
    high = np.vstack([a.max(axis=0) for a in arrays]).max(axis=0)
    width = np.abs(high - low)
    width *= padding
    low -= width
    high += width

    def single_frame(arrays, filename, label=None):

        # create the canvas and divide into matrix
        fig, axes = plt.subplots(
            nrows=num_fields,
            ncols=num_fields,
            figsize=(cell_width * num_fields, cell_height * num_fields))
        fig.subplots_adjust(hspace=cell_margin_y, wspace=cell_margin_x)

        for ax in axes.flat:
            # only show the left and bottom axes ticks and labels
            if ax.is_last_row() and not ax.is_last_col():
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks_position('bottom')
                ax.xaxis.set_major_locator(MaxNLocator(4, prune='both'))
                for tick in ax.xaxis.get_major_ticks():
                    tick.label.set_rotation('vertical')
            else:
                ax.xaxis.set_visible(False)

            if ax.is_first_col() and not ax.is_first_row():
                ax.yaxis.set_visible(True)
                ax.yaxis.set_ticks_position('left')
                ax.yaxis.set_major_locator(MaxNLocator(4, prune='both'))
            else:
                ax.yaxis.set_visible(False)

        # turn off axes frames in upper triangular matrix
        for ix, iy in zip(*np.triu_indices_from(axes, k=0)):
            axes[ix, iy].axis('off')

        levels = np.linspace(0, 1, num_contours + 2)[1:-1]

        # plot the data
        for iy, ix in zip(*np.tril_indices_from(axes, k=-1)):
            ymin = float(low[iy])
            ymax = float(high[iy])
            xmin = float(low[ix])
            xmax = float(high[ix])
            for isample, a in enumerate(arrays):
                hist = Hist2D(
                    num_bins, xmin, xmax,
                    num_bins, ymin, ymax)
                if weights is not None:
                    hist.fill_array(a[:, [ix, iy]], weights[isample])
                else:
                    hist.fill_array(a[:, [ix, iy]])
                # normalize so maximum is 1.0
                _max = hist.GetMaximum()
                if _max != 0:
                    hist /= _max
                r2m.contour(hist,
                    axes=axes[iy, ix],
                    levels=levels,
                    linestyles=sample_lines[isample] if sample_lines else LINES,
                    colors=sample_colors[isample])

        # label the diagonal subplots
        for i, field in enumerate(fields):
            axes[i, i].annotate(field,
                (0.1, 0.2),
                rotation=45,
                xycoords='axes fraction',
                ha='left', va='center')

        # make proxy artists for legend
        lines = []
        for color in sample_colors:
            lines.append(Line2D([0, 0], [0, 0], color=color))

        if sample_names is not None:
            # draw the legend
            leg = fig.legend(lines, sample_names, loc=(0.65, 0.8))
            leg.set_frame_on(False)

        if label is not None:
            axes[0, 0].annotate(label, (0, 1),
                ha='left', va='top',
                xycoords='axes fraction')

        fig.savefig(filename, bbox_inches='tight', dpi=dpi)
        plt.close(fig)

    if animate_field is not None:
        _, ext = os.path.splitext(filename)
        if ext != '.gif':
            raise ValueError(
                "animation is only supported for .gif files")
        field_idx = fields.index(animate_field)
        cuts = np.linspace(
            low[field_idx],
            high[field_idx],
            animate_steps + 1)[:-1]
        gif = GIF()
        temp_dir = tempfile.mkdtemp()
        for i, cut in enumerate(cuts):
            frame_filename = os.path.join(temp_dir, 'frame_{0:d}.png'.format(i))
            label = '{0} > {1:.2f}'.format(animate_field, cut)
            log.info("creating frame for {0} ...".format(label))
            new_arrays = []
            for array in arrays:
                new_arrays.append(array[array[:, field_idx] > cut])
            single_frame(new_arrays,
                filename=frame_filename,
                label=label)
            gif.add_frame(frame_filename)
        gif.write(filename, delay=animate_delay, loop=animate_loop)
        shutil.rmtree(temp_dir)
    else:
        single_frame(arrays, filename=filename)
