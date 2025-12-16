import matplotlib
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
