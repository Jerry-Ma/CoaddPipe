#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2018-07-30 11:44
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
decam.py
"""

import numpy as np
from . import FocalPlaneTilesMixin, ChipLayoutMixin, SkyFootprintMixin


class DECamLayout(FocalPlaneTilesMixin, ChipLayoutMixin, SkyFootprintMixin):

    def __init__(self, binning=1.0):
        self.binning = binning
        self.instru = 'decam'

        # sky footprint
        self.BBOX = (-70. / 60, 70. / 60, -70. / 60, 70. / 60)  # degree
        self.PS = 0.2623 * binning  # pixel scale

        # focal plane tile
        self.NTX = 14
        self.NTY = 14
        self.TW = 2048 / binning
        self.TH = 2048 / binning
        self.TG = 180 / binning  # this value is an approximation

        # chip layout on fp tile
        self.CL = np.array([
                (1, "S29", 1, 2, 4, 6),
                (2, "S31", 1, 2, 8, 10),
                (3, "S25", 2, 3, 3, 5),
                (4, "S26", 2, 3, 5, 7),
                (5, "S27", 2, 3, 7, 9),
                (6, "S28", 2, 3, 9, 11),
                (7, "S20", 3, 4, 2, 4),
                (8, "S21", 3, 4, 4, 6),
                (9, "S22", 3, 4, 6, 8),
                (10, "S23", 3, 4, 8, 10),
                (11, "S24", 3, 4, 10, 12),
                (12, "S14", 4, 5, 1, 3),
                (13, "S15", 4, 5, 3, 5),
                (14, "S16", 4, 5, 5, 7),
                (15, "S17", 4, 5, 7, 9),
                (16, "S18", 4, 5, 9, 11),
                (17, "S19", 4, 5, 11, 13),
                (18, "S8",  5, 6, 1, 3),
                (19, "S9",  5, 6, 3, 5),
                (20, "S10", 5, 6, 5, 7),
                (21, "S11", 5, 6, 7, 9),
                (22, "S12", 5, 6, 9, 11),
                (23, "S13", 5, 6, 11, 13),
                (24, "S1",  6, 7, 0, 2),
                (25, "S2",  6, 7, 2, 4),
                (26, "S3",  6, 7, 4, 6),
                (27, "S4",  6, 7, 6, 8),
                (28, "S5",  6, 7, 8, 10),
                (29, "S6",  6, 7, 10, 12),
                (30, "S7",  6, 7, 12, 14),
                (31, "N1",  7, 8, 0, 2),
                (32, "N2",  7, 8, 2, 4),
                (33, "N3",  7, 8, 4, 6),
                (34, "N4",  7, 8, 6, 8),
                (35, "N5",  7, 8, 8, 10),
                (36, "N6",  7, 8, 10, 12),
                (37, "N7",  7, 8, 12, 14),
                (38, "N8",  8, 9, 1, 3),
                (39, "N9",  8, 9, 3, 5),
                (40, "N10", 8, 9, 5, 7),
                (41, "N11", 8, 9, 7, 9),
                (42, "N12", 8, 9, 9, 11),
                (43, "N13", 8, 9, 11, 13),
                (44, "N14", 9, 10, 1, 3),
                (45, "N15", 9, 10, 3, 5),
                (46, "N16", 9, 10, 5, 7),
                (47, "N17", 9, 10, 7, 9),
                (48, "N18", 9, 10, 9, 11),
                (49, "N19", 9, 10, 11, 13),
                (50, "N20", 10, 11, 2, 4),
                (51, "N21", 10, 11, 4, 6),
                (52, "N22", 10, 11, 6, 8),
                (53, "N23", 10, 11, 8, 10),
                (54, "N24", 10, 11, 10, 12),
                (55, "N25", 11, 12, 3, 5),
                (56, "N26", 11, 12, 5, 7),
                (57, "N27", 11, 12, 7, 9),
                (58, "N28", 11, 12, 9, 11),
                (59, "N29", 12, 13, 4, 6),
                (60, "N31", 12, 13, 8, 10)
                ], dtype=[
                        ('ext', int),
                        ('chip', 'U4'),
                        ('xs', int),
                        ('xe', int),
                        ('ys', int),
                        ('ye', int),
                        ])

        # chip dimensions
        self.CW = 2046 / binning
        self.CH = 4094 / binning
        self.NC = len(self.CL)  # total number of chips (62 - 2)

        self.CX, self.CY = self.layout_extent()
        self.NCX, self.NCY = self.layout_nxy()

    def xy_from_chip(self, chip, x, y):
        '''Return global x and y with given chip name and chip x and y'''
        spec = self.chip(chip)
        return self.xy_from_txy(spec['xs'], spec['ys'], x, y)

    def chip_rect(self, chip):
        '''Return rect (in global x and y: l, r, b, t) of chip'''
        left, bottom = self.xy_from_chip(chip, 0, 0)
        right, top = self.xy_from_chip(chip, self.CW, self.CH)
        return (left, right), (bottom, top)
