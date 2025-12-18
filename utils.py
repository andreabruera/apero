import matplotlib
import numpy
import os

from matplotlib import font_manager

def read_stimuli(categories=False):
    stimuli = list()
    with open(os.path.join('data', 'trials.tsv')) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            phrase = l.split('\t')[0]
            if categories:
                stimuli.append(l.split('\t'))
            else:
                stimuli.append(phrase)
    assert len(stimuli) == 36
    return stimuli

def font_setup(font_folder):
    ### Font setup
    # Using Helvetica as a font
    font_dirs = [font_folder, ]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for p in font_files:
        font_manager.fontManager.addfont(p)
    matplotlib.rcParams['font.family'] = 'Helvetica LT Std'

def load_model(path, triangle=True):
    ft = list()
    with open(path) as i:
        for l in i:
            l = [float(v) for v in l.strip().split('\t')]
            ft.append(l)
    ft = numpy.array(ft)
    assert ft.shape == (36, 36)
    if triangle:
        ft_tri = [ft[x, y] for x in range(36) for y in range(36) if y>x]

        return ft_tri
    else:
        return ft
