# Copyright 2012 the rootpy developers
# distributed under the terms of the GNU General Public License
from __future__ import absolute_import

import subprocess
import os

from . import log; log = log[__name__]

__all__ = [
    'GIF',
]


class GIF(object):

    def __init__(self):

        self.images = []

    def add_frame(self, image):

        self.images.append(image)

    def add_frames(self, images):

        self.images.extend(images)

    def write(self, outname, delay=10, loop=0):

        if not self.images:
            raise RuntimeError(
                "attempting to create an animated GIF without frames")
        name, ext = os.path.splitext(outname)
        if ext != '.gif':
            raise ValueError("output filename must have the .gif extension")
        cmd = [
            'convert',
            '-delay', '{0:d}'.format(delay),
            '-loop', '{0:d}'.format(loop)] + self.images + [outname]
        log.info("creating gif: {0}".format(' '.join(cmd)))
        p = subprocess.Popen(cmd)
        if p.wait():
            raise RuntimeError(
                "failed to create animated GIF: {0}. "
                "Do you have ImageMagick imstalled?".format(outname))

    def clean(self):

        for image in self.images:
            os.unlink(image)
        self.images = []
