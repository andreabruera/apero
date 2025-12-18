import nilearn
import numpy
import os
import pickle
import scipy

from nilearn import image, signal
from scipy import stats
from tqdm import tqdm

from utils import read_stimuli

def read_events(filepath):
    events = list()
    with open(filepath) as i:
        lines = i.readlines()
    header = [val.strip() for val in lines[0].split('\t')]
    for l_i, l in enumerate(lines[1:]):
        split_l = [val.strip() for val in l.split('\t')]
        ### we keep only verbs or nouns
        if 'abs' in split_l[-1] or 'con' in split_l[-1]:
            w = split_l[-2]
        else:
            continue
        ### noun
        if ' ' in w or "'" in w:
            continue
        ### verb
        noun_l = [val.strip() for val in lines[l_i+3].split('\t')]
        assert 'con' in noun_l[-1] or 'abs' in noun_l[-1]
        vp = w + ' ' + noun_l[-2]
        events.append((round(float(split_l[0])), vp))
    return events

stimuli = read_stimuli()

masks = {h : dict() for h in ('left', 'right')}
areas = set()
for root, direc, fz in os.walk(os.path.join('rois', 'masks')):
    for f in fz:
        if 'nii' not in f:
            continue
        hemisphere = f.split('_')[0]
        area = f.split('_')[1]
        areas.add(area)
        mask = nilearn.image.load_img(os.path.join(root, f))
        masks[hemisphere][area] = mask

base_out = os.path.join('brains', 'first_order_mtrx')
os.makedirs(base_out, exist_ok=True)

brains = {h : {a : {s : dict() for s in range(1, 16)} for a in areas} for h in masks.keys()}
mtrxs = {h : {a : {s : numpy.zeros(shape=(25, 36, 36)) for s in range(1, 16)} for a in areas} for h in masks.keys()}
sel_mtrxs = {h : {a : {s : numpy.zeros(shape=(25, 36, 36)) for s in range(1, 16)} for a in areas} for h in masks.keys()}

dim = {h : dict() for h in masks.keys()}

for sub in tqdm(range(1, 16)):
    folder = '../../neuroscience/dot_lunch_fast/derivatives/sub-{:02}/ses-mri/func'.format(sub)
    all_files = os.listdir(folder)
    for run in tqdm(range(1, 6)):
        run_fs = [f for f in all_files if 'run-{:02}'.format(run) in f]
        assert len(run_fs) == 2
        run_events = [f for f in run_fs if 'tsv' in f]
        assert len(run_events) == 1
        events = read_events(os.path.join(folder, run_events[0]))
        run_imgs = [f for f in run_fs if 'nii' in f]
        assert len(run_imgs) == 1
        img = nilearn.image.load_img(os.path.join(folder, run_imgs[0]))
        for hemi, hemi_masks in masks.items():
            for area, mask in hemi_masks.items():
                res_mask = nilearn.image.resample_to_img( 
                                                         mask, 
                                                         img, 
                                                         interpolation='nearest',
                                                         force_resample=True,
                                                         copy_header=True,
                                                         )
                bin_mask = nilearn.image.binarize_img(
                                                      res_mask, 
                                                      threshold=0.9,
                                                      copy_header=True,
                                                      two_sided=False,
                                                      )
                ### applying mask
                masked_img = nilearn.masking.apply_mask(img, bin_mask)
                ### cleaning the signal
                masked_img = nilearn.signal.clean(
                                                  masked_img,
                                                  standardize='zscore_sample',
                                                  t_r=1,
                                                  )
                dim[hemi][area] = masked_img.shape[-1]
                for start, event in events:
                    try:
                        brains[hemi][area][sub][event].append(masked_img[start-5:min(start+20, masked_img.shape[0])])
                    except KeyError:
                        brains[hemi][area][sub][event] = [masked_img[start-5:min(start+20, masked_img.shape[0]), :]]
    for h, h_data in brains.items():
        for a, a_data in h_data.items():
            curr_sub_data = brains[h][a][sub]
            assert len(curr_sub_data.keys()) == 36
            avgs = numpy.zeros(shape=(36, 25, dim[h][a]))
            for k, v in curr_sub_data.items():
                assert len(v) == 5
                for vec in v:
                    assert vec.shape == (25, dim[h][a])
            stds = numpy.zeros(shape=(36, 25, dim[h][a]))
            for phrase_i, phrase in enumerate(stimuli):
                vec = numpy.average(curr_sub_data[phrase], axis=0)
                assert vec.shape == (25, dim[h][a])
                avgs[phrase_i] = vec
                std = numpy.std(curr_sub_data[phrase], axis=0)
                stds[phrase_i] = std
            ### transforming to ranks
            ranks = scipy.stats.rankdata(avgs, axis=2)
            ### correlations
            for t in range(25):
                corrs = numpy.corrcoef(ranks[:, t, :])
                assert corrs.shape == (36, 36)
                mtrxs[h][a][sub][t] = corrs
            ### with feature selection
            avg_stds = numpy.average(stds, axis=0)
            assert avg_stds.shape == (25, dim[h][a])
            rank_stds = scipy.stats.rankdata(avg_stds, method='ordinal', axis=1)
            sel_avgs = numpy.zeros(shape=(36, 25, 100))
            for phrase_i, phrase in enumerate(stimuli):
                phrase_ts = list()
                for t in range(25):
                    chosen_dims = [_ for _, i in enumerate(rank_stds[t]) if i <= 100]
                    chosen_vec = numpy.array(curr_sub_data[phrase])[:, t, chosen_dims]
                    try:
                        assert chosen_vec.shape == (5, 100)
                    except AssertionError:
                        import pdb; pdb.set_trace()
                    vec = numpy.average(chosen_vec, axis=0)
                    phrase_ts.append(vec)
                vec = numpy.array(phrase_ts)
                assert vec.shape == (25, 100)
                sel_avgs[phrase_i] = vec
            ### transforming to ranks
            sel_ranks = scipy.stats.rankdata(sel_avgs, axis=2)
            ### correlations
            for t in range(25):
                corrs = numpy.corrcoef(sel_ranks[:, t, :])
                assert corrs.shape == (36, 36)
                sel_mtrxs[h][a][sub][t] = corrs

with open(os.path.join(base_out, 'rois.pkl'), 'wb') as o:
    pickle.dump(mtrxs, o)
with open(os.path.join(base_out, 'rois_selected-features.pkl'), 'wb') as o:
    pickle.dump(sel_mtrxs, o)
base_out = os.path.join('brains', 'bold')
os.makedirs(base_out, exist_ok=True)
with open(os.path.join(base_out, 'rois.pkl'), 'wb') as o:
    pickle.dump(brains, o)
