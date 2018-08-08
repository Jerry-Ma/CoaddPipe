#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2018-07-30 15:35
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
__init__.py
"""

from astropy.io import fits
import numpy as np


class FocalPlaneTilesMixin(object):
    """
    Setup a tiled description of the focal plane, on which multiple
    Chips could be placed.

    Require the following attributes:
        NTX  # number of tile on x
        NTY  # number of tile on y
        TW   # width
        TH   # height
        TG   # gap
    """

    def xy_from_txy(self, tx, ty, x, y):
        '''Return global x and y with given tile indices and tile x and y'''
        if tx >= self.NTX or tx < 0 or ty >= self.NTY or ty < 0:
            raise ValueError("invalid tile indices")
        xx = tx * (self.TW + self.TG) + x
        yy = ty * (self.TH + self.TG) + y
        return xx, yy

    def tile_rect(self, tx, ty):
        '''Return rect in global x and y of (l, r, b, t) of tile indices'''
        if tx >= self.NTX or tx < 0 or ty >= self.NTY or ty < 0:
            raise ValueError("invalid tile indices")
        left, bottom = self.xy_from_txy(tx, ty, 0, 0)
        right, top = self.xy_from_txy(tx, ty, self.TW, self.TH)
        return (left, right), (bottom, top)

    def tile_bins(self):
        '''Return two list of tuples, for x and y direction, respectively.
        Each tuple in each list is the edge global x or y
        of the tile'''
        xbin = []
        for t in range(self.NTX):
            rect = self.tile_rect(t, 0)
            xbin.append(rect[0])
        ybin = []
        for t in range(self.NTY):
            ybin.append(rect[1])
        return xbin, ybin


class ChipLayoutMixin(object):
    """
    Handle chips spec

    Require:
        CL:
            ext: extension index
            chip: extension name
            xs: x start on fp tile
            xe: x end on fp tile
            ys: y start on fp tile
            ye: y end on fp tile
    """

    def ext(self, ext):
        return self.CL[self.CL['ext'] == ext][0]

    def chip(self, chip):
        return self.CL[self.CL['chip'] == chip][0]

    def chips(self):
        return self.CL['chip']

    def layout_extent(self):
        return (min(self.CL['xs']), max(self.CL['xe'])), (
                min(self.CL['ys']), max(self.CL['ye']))

    def layout_nxy(self):
        (l, r), (b, t) = self.layout_extent()
        return r - l, t - b

    def fxy(self, fx, fy, key=None):
        chips = self.CL[
                (fx >= self.CL['xs']) & (fx < self.CL['xe'])
                & (fy >= self.CL['ys']) & (fy < self.CL['ye'])
                ]
        if len(chips) == 0:
            return None
        elif key is None:
            return chips[0]
        else:
            return chips[0][key]

    def enumerate(self, hdulist=None):
        for spec in self.CL:
            if hdulist is not None:
                yield spec['ext'], spec['chip'], hdulist[spec['ext']]
            else:
                yield spec['ext'], spec['chip']


class SkyFootprintMixin(object):
    """
    Handle sky extent

    Require:
        BBOX
        PS
    """
    def sky_bbox(self, center=None):
        '''return range of ra and dec for image'''
        if center is None:
            ra = dec = 0
        else:
            ra, dec = center
        cosdec = np.cos(dec * np.pi / 180.)
        w, e, s, n = self.BBOX
        return ra + w / cosdec, ra + e / cosdec, dec + s, dec + n

    def sky_cbox(self, center=None):
        w, e, s, n = self.sky_bbox(center=center)
        cra = (w + e) * 0.5
        cdec = (s + n) * 0.5
        dra = (e - w) * np.cos(cdec * np.pi / 180.)
        ddec = (n - s)
        return cra, cdec, dra, ddec


def get_layout(image=None, binning=1.0, instru=None):

    if image is not None:
        if isinstance(image, fits.HDUList):
            instru = image[0].header['INSTRUME'].lower()
        else:
            with fits.open(image, memmap=True) as hdulist:
                instru = hdulist[0].header['INSTRUME'].lower()
    if instru == '5odi':
        from .wiyn import ODILayout5ODI
        return ODILayout5ODI(binning=binning)
    elif instru == 'podi':
        from .wiyn import ODILayoutPODI
        return ODILayoutPODI(binning=binning)
    elif instru == 'decam':
        from .decam import DECamLayout
        return DECamLayout(binning=binning)
    else:
        raise ValueError("unknown instrument {}".format(instru))
