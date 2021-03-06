#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Create Date    :  2016-07-30 23:04
# Git Repo       :  https://github.com/Jerry-Ma
# Email Address  :  jerry.ma.nk@gmail.com
"""
prep_masking.py

The script is called to mask-out selected chips of the input image
according to the specified jobfile table.

The program looks into the "mask_chips" column, determine
what chips are masked (filled with NANs).

In addition to the selected chips, a set of bad pixel masks
are also applied to the images. The bad pixel masks are stored
in bpm/ folder aside the location of this script.

Inputs
------
image: fits image
    The input fits image to be masked. This filename should contain
    OBSID to let the jobfile know which entry to look for mask_chips info.

jobfile: ASCII table
    The ASCII table generated from running `postcalib init` and approved
    (edited) by the user if necessary. It should contain column "mask_chips"
    and "OBSID".
    The mask_chips column is used to get which OTAs are to be masked, and
    the OBSID is used to locate the entry to which the input image corresponds

Outputs
-------
out_image: fits images
    The images with undesired pixels/OTAs masked as NAN.
"""


import os
import re
import sys
import glob

import numpy as np
from astropy.io import fits
from astropy.table import Table
# from functools import partial
import itertools
# from multiprocessing import Pool, cpu_count
# import multiprocessing
# import multiprocessing.pool
# from multiprocessing import cpu_count
# from concurrent.futures import ProcessPoolExecutor as Pool
import subprocess
# import shutil

from ..instruments import get_layout
from ..apus.common import get_log_func, touch_file
from .. import qa
# from postcalib.utils import mp_traceback


def main(*args, **kwargs):
    log = get_log_func(default_level='debug', **kwargs)
    if not args:
        args = sys.argv[1:]
    image, jobfile, outname = args

    job_table = Table.read(jobfile, format='ascii.commented_header')
    # job_table.pprint(max_width=None, max_lines=-1)

    # locate the image entry
    entry = [e for e in job_table if e['OBSID'] in image]

    if len(entry) != 1:
        raise RuntimeError(
                "inconsistent job table and jobdir content."
                " Re-run `select images` task to clean up the jobdir")
    entry = entry[0]

    logid = "#{} {}".format(entry['numid'], image)
    log("masking {}".format(logid))
    # uncompress the data if compressed
    # shutil.copyfile(image, outname)
    maskfile = os.path.join(
            os.path.dirname(image),
            os.path.basename(image).replace("orig_", 'instcaldqmask_'))
    wtfile = os.path.join(
            os.path.dirname(image),
            os.path.basename(image).replace("orig_", 'instcalwtmap_'))
    if not os.path.exists(maskfile):
        maskfile = None
    if not os.path.exists(wtfile):
        wtfile = None
    files_unpack = [image, maskfile, wtfile]
    for i, (f, o) in enumerate(zip(files_unpack, [
            outname,
            outname.replace(".fits", '.bpm.fits'),
            outname.replace(".fits", '.wht.fits'),
            ])):
        if f is None:
            if i == 0:
                filename = image  # work on original data
            elif i == 1:
                bpmfile = maskfile
            elif i == 2:
                whtfile = wtfile
        elif f.endswith(".fz"):
            # unpack to tmp file, and mv after finish
            tmpname = o + ".tmp"
            # need to make sure no tmp file is there before run funpack
            if os.path.exists(tmpname):
                os.remove(tmpname)
            if subprocess.check_call(
                    [kwargs['funpack_cmd'], '-O', tmpname, f]) == 0:
                os.rename(tmpname, o)
                log("unpacked {}".format(f))
            else:
                raise RuntimeError('fail to unpack {}'.format(f))
            if i == 0:
                filename = o  # work on uncompressed data
            elif i == 1:
                bpmfile = o
            elif i == 2:
                whtfile = o
        else:
            log("fits is unpack already")
            if i == 0:
                filename = image  # work on original data
            elif i == 1:
                bpmfile = maskfile
            elif i == 2:
                whtfile = wtfile

    with fits.open(filename, memmap=True) as hdulist:
        layout = get_layout(hdulist)
        chips = get_mask_chips(str(entry['mask_chips']), layout)
        log("mask chips {} for {}".format(chips, logid))
        hdulist = apply_mask(
                hdulist, layout, chips, bpmdir=kwargs['bpmdir'],
                bpmfile=bpmfile
                )
        log("write masked sci extentions {}".format(outname))
        sciexts = []
        for ext, _, _, in layout.enumerate(hdulist):
            sciexts.append(ext)
        scilist = [hdulist[0], ]
        asslist = [hdulist[0], ]
        for e, hdu in enumerate(hdulist):
            if e == 0:
                continue
            if e in sciexts:
                scilist.append(hdu)
            else:
                asslist.append(hdu)
        scilist = fits.HDUList(scilist)
        scilist.writeto(outname, overwrite=True)
        out_assoc = outname.rsplit(".fits", 1)[0] + ".assoc"
        log("write associate extentions {}".format(out_assoc))
        asslist = fits.HDUList(asslist)
        asslist.writeto(out_assoc, overwrite=True)
        pr = qa.create_preview(
                hdulist=hdulist, filename=outname, delete_data=True)
        pr.save()
        del scilist
        del asslist
    if whtfile is not None:
        outname = outname.replace(".fits", '.wht.fits')
        with fits.open(whtfile, memmap=True) as hdulist:
            layout = get_layout(hdulist)
            chips = get_mask_chips(str(entry['mask_chips']), layout)
            log("mask chips {} for {}".format(chips, logid))
            hdulist = apply_mask(
                    hdulist, layout, chips, bpmdir=kwargs['bpmdir'],
                    bpmfile=bpmfile
                    )
            log("write masked wht extentions {}".format(outname))
            sciexts = []
            for ext, _, _, in layout.enumerate(hdulist):
                sciexts.append(ext)
            scilist = [hdulist[0], ]
            asslist = [hdulist[0], ]
            for e, hdu in enumerate(hdulist):
                if e == 0:
                    continue
                if e in sciexts:
                    scilist.append(hdu)
                else:
                    asslist.append(hdu)
            scilist = fits.HDUList(scilist)
            scilist.writeto(outname, overwrite=True)
            out_assoc = outname.rsplit(".fits", 1)[0] + ".assoc"
            log("write associate extentions {}".format(out_assoc))
            asslist = fits.HDUList(asslist)
            asslist.writeto(out_assoc, overwrite=True)
            pr = qa.create_preview(
                    hdulist=hdulist, filename=outname, delete_data=True)
            pr.save()
            del scilist
            del asslist


def select_images(jobfile, jobdir, checkfile, **kwargs):
    log = get_log_func(default_level='debug', **kwargs)

    job_table = Table.read(jobfile, format='ascii.commented_header')
    job_table.pprint(max_width=None, max_lines=-1)

    # cases:
    # no checker_table
    #   entries in job_table, add/remove depending on otas
    # has checker_table
    #   entries add could be skip if the otas are the same
    #   entries remove should always be done
    #   may have additional entries to be removed

    if not os.path.exists(checkfile):
        checker_table = []
    else:
        checker_table = Table.read(
                checkfile, format='ascii.commented_header')

    def rm_entry(outname, outglob):
        if os.path.islink(outname):
            log("unlink {}".format(outname))
            os.unlink(outname)
            success = True
        else:
            success = False
        # purge any existing data
        for f in glob.glob(outglob):
            log("purged {}".format(f))
            if os.path.islink(f):
                os.unlink(f)
            else:
                os.remove(f)
        return success

    add_count = skip_count = rm_count = touch_count = 0
    for entry in job_table:
        # naming
        inname = entry['filename']
        # print(inname, kwargs['reg_orig'])
        parsed_filename = re.match(
                kwargs['reg_orig'], os.path.basename(inname)
                ).groupdict()
        assglob = os.path.join(os.path.dirname(inname), "*_{obsid}_*".format(
            **parsed_filename))
        outname = os.path.join(
            jobdir, kwargs['fmt_inputs'].format(
                # featgrp=entry['feature_group'],
                # photgrp=entry['phot_group'],
                # mscgrp=entry['mosaic_group'],
                **parsed_filename))
        outglob = os.path.join(jobdir, "*_{obsid}_*".format(**parsed_filename))
        # linksrc = os.path.relpath(inname, jobdir)
        # linksrc = os.path.relpath(os.path.realpath(inname),
        #                           os.path.realpath(jobdir))
        # otas
        layout = get_layout(instru=entry['INSTRUME'].lower())
        chips = get_mask_chips(str(entry['mask_chips']), layout=layout)
        # handle remove of outname and purge related files
        if len(chips) == layout.NC:
            success = rm_entry(outname, outglob)
            if success:
                rm_count += 1
        else:  # adding
            checker_entry = [e for e in checker_table
                             if e['filename'] == entry['filename']]
            if os.path.exists(outname) and len(checker_entry) > 0 and \
                    set(chips) == set(get_mask_chips(
                        str(checker_entry[0]['mask_chips']), layout=layout)):
                log("skip {}".format(outname))
                skip_count += 1
                continue
            if os.path.exists(outname):
                log("touch target exist {}".format(outname))
                touch_file(outname)
                touch_count += 1
            else:
                for name in glob.glob(assglob):
                    linksrc = os.path.abspath(os.path.realpath(name))
                    if name == inname:
                        log("link {} -> {}".format(linksrc, outname))
                        if os.path.islink(outname):
                            os.unlink(outname)
                        os.symlink(linksrc, outname)
                    else:
                        print(name)
                        print(kwargs['reg_orig'])
                        pname = re.match(
                                kwargs['reg_orig'], os.path.basename(name)
                                )
                        if pname is None:
                            continue
                        else:
                            pname = pname.groupdict()
                        oname = os.path.join(
                            jobdir, kwargs['fmt_inputs'].replace(
                                'orig_', '{imflag}_').format(
                                # featgrp=entry['feature_group'],
                                # photgrp=entry['phot_group'],
                                # mscgrp=entry['mosaic_group'],
                                **pname))
                        log("link assoc {} -> {}".format(linksrc, oname))
                        if os.path.islink(oname):
                            os.unlink(oname)
                        os.symlink(linksrc, oname)
                add_count += 1
    # remove addition non existing files
    for entry in checker_table:
        inname = entry['filename']
        parsed_filename = re.match(
                kwargs['reg_orig'], os.path.basename(inname)
                ).groupdict()
        outname = os.path.join(
            jobdir, kwargs['fmt_inputs'].format(**parsed_filename))
        outglob = os.path.join(jobdir, "*_{obsid}_*".format(**parsed_filename))
        job_entry = [e for e in job_table
                     if e['filename'] == entry['filename']]
        if len(job_entry) == 0:
            success = rm_entry(outname, outglob)
            if success:
                rm_count += 1
    log("entries + {}, - {}, touched {}, skipped {}".format(
        add_count, rm_count, touch_count, skip_count))

    job_table.write(checkfile, format='ascii.commented_header')


def apply_mask(hdulist, layout, mask_chips, bpmdir=None, bpmfile=None):
    # look for badpixel mask in bmp dir aside this script
    _bpmdir = os.path.join(os.path.dirname(__file__), 'bpm')
    if layout.instru == '5odi':
        _bpmdir = os.path.join(_bpmdir, 'odi_5x6')
    elif layout.instru == 'podi':
        _bpmdir = os.path.join(_bpmdir, 'podi')
    elif layout.instru == 'decam':
        _bpmdir = None
    else:
        raise ValueError('ODI instru {0} not recognized'
                         .format(layout.instru))
    # handle bpmfile
    if bpmfile is not None and os.path.exists(bpmfile):
        print("apply mask file {}".format(os.path.basename(bpmfile)))
        masklist = fits.open(bpmfile, memmap=True)
    else:
        masklist = None
    for i, (ext, chip, hdu) in enumerate(layout.enumerate(hdulist)):
        # print("work on ext {} otaxy {}".format(ext, otaxy))
        if chip in mask_chips:
            hdu.data[:, :] = np.nan
            continue
        if _bpmdir is not None:
            bpm_files = [os.path.join(_bpmdir, 'bpm_xy{0}.reg'.format(chip)), ]
        else:
            bpm_files = []
        if bpmdir is not None:
            bpm_files.extend(glob.glob(
                os.path.join(bpmdir, 'bpm_xy{}.reg'.format(chip))))
        bpm = []
        for bpm_file in bpm_files:
            # print(bpm_file)
            with open(bpm_file, 'r') as fo:
                for ln in fo.readlines():
                    rect = re.match(r'box\(([0-9+-., ]+)\)', ln.strip())
                    if rect is not None:
                        rect = list(map(float, rect.group(1).split(',')))
                        # print "box from bpm: {0}".format(rect)
                        bpm.append((
                            max([rect[0] - rect[2] * 0.5, 0]),
                            rect[0] + rect[2] * 0.5,
                            max([rect[1] - rect[3] * 0.5, 0]),
                            rect[1] + rect[3] * 0.5))
                    else:
                        if ln.startswith("box"):
                            raise Exception("should not happen")
                        continue
        data = hdu.data[:, :]
        for box in bpm:
            l, r, b, t = (int(box[0]), int(box[1]) + 1,
                          int(box[2]), int(box[3]) + 1)
            data[b:t, l:r] = np.nan
        if masklist is not None:
            data[masklist[ext].data > 0] = np.nan
        hdulist[ext].data = data
    return hdulist


def parse_mask_chips(code):
    codes = list(map(str.strip, code.split(',')))
    chips = []
    for code in codes:
        # "--" is the astropy mask chars
        if code == "" or code == "--" or len(code) > 2:
            continue
        elif code == "*":
            code = "**"
        x, y = code
        xs = map(str, range(10)) if x == "*" else [x, ]
        ys = map(str, range(10)) if y == "*" else [y, ]
        chips.extend([i + j for i, j in itertools.product(xs, ys)])
    # TODO add mask of decam chips
    return set(map(int, chips))


def get_mask_chips(code, layout):
    chips = parse_mask_chips(code).intersection(set(layout.CL['ext']))
    return sorted(list(chips))


if __name__ == "__main__":
    main()


# def main(*args, **kwargs):
#     log = get_log_func(default_level='debug', **kwargs)
#     if not args:
#         args = sys.argv[1:]
#     jobfile, jobdir, checkfile = args

#     job_table = Table.read(jobfile, format='ascii.commented_header')
#     job_table.pprint(max_width=None, max_lines=-1)

#     # get different entries
#     if os.path.exists(checkfile):
#         checker_table = Table.read(
#                 checkfile, format='ascii.commented_header')
#         diff = []
#         for entry in job_table:
#             checker_entry = checker_table[
#                     checker_table['filename'] == entry['filename']]
#             if len(checker_entry) == 0:
#                 diff.append(entry)
#             elif len(checker_entry) == 1:
#                 checker_entry = checker_entry[0]
#                 if entry['mask_otas'] != checker_entry['mask_otas']:
#                     diff.append(entry)
#     else:
#         diff = job_table
#     log("updating {} entries".format(len(diff)))

#     # mp_worker(job_table[0], jobdir=jobdir, kwargs=kwargs)
#     pool = Pool(cpu_count())
#     # with Pool(1) as pool:
#     pool.map_async(
#             partial(mp_worker, jobdir=jobdir, kwargs=kwargs),
#             diff).get(9999999)
#     job_table.write(checkfile, format='ascii.commented_header')


# def main(*args, **kwargs):
#     log = get_log_func(default_level='debug', **kwargs)
#     if not args:
#         args = sys.argv[1:]
#     (image, jobfile, checkfile), outname = args

#     job_table = Table.read(jobfile, format='ascii.commented_header')
#     job_table.pprint(max_width=None, max_lines=-1)

#     checker_table = Table.read(
#                 checkfile, format='ascii.commented_header')

#     # locate the image entry
#     entry = [e for e in job_table if e['OBSID'] in image]
#     if len(entry) != 1:
#         raise RuntimeError(
#                 "inconsistent job table and jobdir content."
#                 " Re-run `select images` task to clean up the jobdir")
#     entry = entry[0]
#     checker_entry = checker_table[
#             checker_table['filename'] == entry['filename']]
#     # determine if we need run the mask

#     def is_changing(entry, checker_entry):
#         layout = get_layout(instru=entry['INSTRUME'])
#         otas1 = get_mask_otas(str(entry['mask_otas']), layout=layout)
#         otas2 = get_mask_otas(str(checker_entry['mask_otas']), layout=layout)
#         return set(otas1) != set(otas2)

#     logid = "#{} {}".format(entry['numid'], image)
#     if len(checker_entry) == 0 or is_changing(entry, checker_entry):
#         log("updating {}".format(logid))
#         # uncompress the data if compressed
#         shutil.copyfile(image, outname)
#         subprocess.check_call([kwargs['funpack_cmd'], '-F', outname])
#         with fits.open(outname, memmap=True) as hdulist:
#             layout = get_layout(hdulist)
#             otas = get_mask_otas(str(entry['mask_otas']), layout)
#             log("mask otas {} for {}".format(otas, logid))
#             hdulist = apply_mask(hdulist, layout, otas)
#             log("write masked sci extentions {}".format(outname))
#             hdulist[:layout.n_ota + 1].writeto(outname, overwrite=True)
#             out_assoc = outname.rstrip(".fits") + ".assoc"
#             log("write associate extentions {}".format(out_assoc))
#             fits.HDUList(hdulist[:1] + hdulist[layout.n_ota + 1:]).writeto(
#                     out_assoc, overwrite=True)
#             pr = qa.create_preview(
#                     hdulist=hdulist, filename=outname, delete_data=True)
#             pr.save()
#     else:
#         log("no change of image {}".format(logid))


# @mp_traceback
# def mp_worker(entry, jobdir, kwargs):
#     log = get_log_func(default_level='debug', **kwargs)
#     logid = "#{} {}".format(entry['numid'], entry['filename'])
#     log("working on {}".format(logid))

#     # compose output filename
#     inname = entry['filename']
#     parsed_filename = re.match(
#             kwargs['reg_inputs'], os.path.basename(inname)
#             ).groupdict()
#     outname = os.path.join(
#             jobdir, kwargs['fmt_masked'].format(**parsed_filename))
#     # uncompress the data if compressed
#     shutil.copyfile(inname, outname)
#     subprocess.check_call([kwargs['funpack_cmd'], '-F', outname])

#     with fits.open(outname, memmap=True) as hdulist:
#         layout = get_layout(hdulist)
#         otas = get_mask_otas(str(entry['mask_otas']), layout)
#         if len(otas) == layout.n_ota:
#             log("skipped for {}".format(logid))
#             # here we purge all stuff with the image osbid
#             for f in glob.glob(
#                     os.path.join(jobdir, "*_{obsid}_*".format(
#                     **parsed_filename))):
#                 log("purged {}".format(f))
#                 os.remove(f)
#         else:
#             log("mask otas {} for {}".format(otas, logid))
#             hdulist = apply_mask(hdulist, layout, otas)

#             log("write masked sci extentions {}".format(outname))
#             hdulist[:layout.n_ota + 1].writeto(outname, overwrite=True)
#             out_assoc = outname.rstrip(".fits") + ".assoc"
#             log("write associate extentions {}".format(out_assoc))
#             fits.HDUList(hdulist[:1] + hdulist[layout.n_ota + 1:]).writeto(
#                     out_assoc, overwrite=True)
#             pr = qa.create_preview(
#                     hdulist=hdulist, filename=outname, delete_data=True)
#             pr.save()
