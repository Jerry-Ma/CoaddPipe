#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2017-08-14 14:54
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
qa.py
"""


from .instruments import get_layout
import os
import logging
import warnings
from astropy.visualization import ZScaleInterval  # , PercentileInterval
from astropy.visualization.mpl_normalize import ImageNormalize
import numpy as np
from astropy.io import fits
# from astropy.stats import sigma_clip  # , mad_std
# import itertools

import matplotlib
matplotlib.use("agg")
# import matplotlib.image as mimg
import matplotlib.pyplot as plt  # noqa: E402


class Preview(object):
    """
    A thin wrapper for previewing a fits image
    """
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def save(self, name=None):

        if name is None:
            savename = self.default_savename
        elif os.path.isdir(name):
            savename = os.path.join(name, os.path.basename(
                self.default_savename))
        else:
            savename = name
        self.fig.savefig(savename, pad_inches=0.0, bbox_inches='tight')
        self.logger.info('save preview to {0}'.format(savename))

    def show(self):
        # plt.show()
        raise NotImplementedError()


def create_preview(hdulist=None, binning=8, filename=None, delete_data=True):
    logger = logging.getLogger("qa.preview")

    if hdulist is None and filename is not None:
        hdulist = fits.open(filename, memmap=True)
        close_after = True
    elif hdulist is not None:
        if filename is None:
            filename = hdulist.filename()
        close_after = False
    else:
        raise ValueError("there has to be at least one of hdulist or filename")
    filenamebase = os.path.basename(filename).rsplit(".fz", 1)[0].rsplit(
            ".fits", 1)[0]
    dirname = os.path.dirname(filename)

    logger.info("create preview for {}".format(filename))
    layout = get_layout(hdulist, binning=binning)
    logger.info("instrument {}, number of chips {}, preview binning {}".format(
        layout.instru, layout.NC, binning))

    # binning mosaic
    (_, size_x), (_, size_y) = layout.tile_rect(
            layout.NCX - 1, layout.NCY - 1)
    size_x, size_y = map(int, (size_x, size_y))
    preview_data = np.empty((size_y, size_x), dtype='f') * np.nan
    binned_data = []
    l0, b0 = layout.xy_from_txy(
            layout.CX[0], layout.CY[0],
            0, 0)  # offset to left bottom corner
    for _, chip, hdu in layout.enumerate(hdulist):
        bin_shape = tuple(map(int, (layout.CH, binning, layout.CW, binning)))
        data_shape = (
                int(bin_shape[0] * bin_shape[1]),
                int(bin_shape[1] * bin_shape[2]))
        data = np.empty(data_shape) * np.nan
        copy_shape = (
                min(data.shape[0], hdu.data.shape[0]),
                min(data.shape[1], hdu.data.shape[1]))
        data[:copy_shape[0], :copy_shape[1]] = hdu.data[
                :copy_shape[0], :copy_shape[1]]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            binned = np.nanmean(
                    np.nanmean(
                        np.reshape(data, bin_shape), axis=-1), axis=1)
        # binned[0:100, 0:100] = np.nan
        interval = ZScaleInterval()
        # interval = PercentileInterval(99)
        try:
            vmin, vmax = interval.get_limits(binned)
        except IndexError:
            vmin, vmax = np.min(binned), np.max(binned)
        norm = ImageNormalize(vmin=vmin, vmax=vmax)

        l, b = layout.xy_from_chip(chip, 0, 0)
        l = int(l - l0)  # noqa: E741
        b = int(b - b0)
        preview_data[b:b + binned.shape[0], l:l + binned.shape[1]] = norm(
                binned)
        binned_data.append(binned)
        if hdulist._file.memmap and delete_data:
            del hdu.data  # possibly free some memory
    binned_data = np.dstack(binned_data)

    # guide_otas = find_guide_otas(
    #         binned_data, layout, thresh=10, logger=logger)

    # preview figure
    fig = plt.figure(figsize=(2 * layout.NCX, 2 * layout.NCY))
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(preview_data, vmin=0, vmax=1, origin='lower', aspect=1)
    tick_x, tick_y = layout.tile_bins()
    tick_x = [(x[0] + x[1]) * 0.5 for x in tick_x][:layout.NCX]
    tick_y = [(y[0] + y[1]) * 0.5 for y in tick_y][:layout.NCY]
    ax.yaxis.set_ticks(tick_y)
    ax.yaxis.set_ticklabels(['Y{0}'.format(j) for j
                             in range(*layout.CY)])
    ax.xaxis.set_ticks(tick_x)
    ax.xaxis.set_ticklabels(['X{0}'.format(j) for j
                             in range(*layout.CX)])
    ax.set_title(filenamebase)

    default_savename = os.path.join(dirname, filenamebase + '.png')
    pr = Preview(
            fig=fig, ax=ax, logger=logger, default_savename=default_savename,
            binning=binning, binned_data=binned_data,
            mask_chips=(),
            preview_data=preview_data,
            layout=layout)
    if close_after:
        hdulist.close()
    return pr


# if __name__ == "__main__":
#     import sys
#     # from astropy.io import fits
#     with fits.open(sys.argv[1]) as hdulist:
#         create_preview(hdulist)
