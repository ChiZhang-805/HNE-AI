"""Semantic masks for the fixed, dimension-invariant HNE representation."""
import numpy as np


def coordinate_mask(name):
    keep = np.ones(296, dtype=bool)
    statistics = {'HNE-no-amplitude': (2, 3), 'HNE-no-location': (0, 1, 5),
                  'HNE-no-memory': (6, 7), 'HNE-no-tail': (4,)}
    if name in statistics:
        for scale in range(3):
            for statistic in statistics[name]:
                for offset in (0, 132):
                    start = offset + (scale*8+statistic)*4
                    keep[start:start+4] = False
                keep[236+scale*8+statistic] = False
    elif name in ('HNE-scale8', 'HNE-scale16', 'HNE-scale32'):
        chosen = (8, 16, 32).index(int(name.removeprefix('HNE-scale')))
        for scale in range(3):
            if scale == chosen:
                continue
            for offset, width in ((0,32), (96,12), (132,32), (236,8), (260,12)):
                keep[offset+scale*width:offset+(scale+1)*width] = False
    elif name == 'HNE-context-append':
        keep[132:236] = False
        keep[260:296] = False
    return keep
