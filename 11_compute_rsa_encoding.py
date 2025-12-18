import nilearn
import numpy
import os
import pickle
import random
import scipy

from matplotlib import pyplot
from nilearn import glm
from tqdm import tqdm

from utils import font_setup, load_model, read_stimuli

### BOLD response for 12 seconds after stimulus

bold_hrf = glm.first_level.glover_hrf(t_r=1., oversampling=1.)[:20]**3
### adding 0. before stimulus appearance
bold_hrf = numpy.hstack([[0., 0., 0., 0., 0.], bold_hrf])

stimuli = read_stimuli()

### train-test splits (approx 80%-20%, repeated 20 times)
test_splits = [random.sample(range(len(stimuli)), k=7) for _ in range(20)]
train_splits = [[_ for _ in range(36) if _ not in test] for test in test_splits]

models = {
         ### vectors
         'conceptnet' : os.path.join('models', 'first_level_mtrx', 'cosine', 'conceptnet_verbnoun.tsv'),
         'fasttext-phrase' : os.path.join('models', 'first_level_mtrx', 'cosine', 'fasttext_verbnoun.tsv'),
         'coocs-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs-wac.tsv'),
         'coocs10000-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs10000-wac.tsv'),
         'coocs10000-cc100' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs10000-cc100.tsv'),
         ### surprisal
         'surprisal-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'surprisal-wac.tsv'),
         ### morphological
         'phrase-length' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'phrase-levenshtein_phrase.tsv'),
         'phrase-levenshtein' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'phrase-length_phrase.tsv'),
         'old20-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'old20-wac.tsv'),
         'freqs-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'freqs-wac.tsv'),
         ### categorical
         'food-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'food-nofood-categorical.tsv'),
         'composition-type-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'composition-type-categorical.tsv'),
         'concreteness-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'abstract-concrete-categorical.tsv'),
         ### conc / imag / fam
         'imageability' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'imageability_phrase.tsv'),
         'concreteness-continuous' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'concreteness_phrase.tsv'),
         'familiarity' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'familiarity_phrase.tsv'),
         }

colors = ['mediumorchid',
                'plum',
                'thistle',
                'mediumaquamarine',
                'olive',
                'peru',
                'darkkhaki',
                'sandybrown',
                'khaki',
                'peachpuff',
                'paleturquoise',
                'mediumturquoise',
                'lightskyblue',
                'teal',
                'lightsalmon',
               'darkgray',
                ]
colors = random.sample(colors, k=len(colors))

results = dict()

for metric in ['spearman', 'pairwise']:
    results[metric] = dict()
    for model, model_path in models.items():
        results[metric][model] = dict()
        print(model)

        model_mtrx = load_model(model_path, triangle=False)
        with tqdm() as counter:
            for case in (
                         'rois', 
                         #'rois_selected-features',
                         ):
                out = os.path.join('plots', 'time_resolved', 'rsa_encoding', metric, case)
                os.makedirs(out, exist_ok=True)

                with open(os.path.join('brains', 'bold', '{}.pkl'.format(case)), 'rb') as i:
                    brains = pickle.load(i)
                for hemi, hemi_data in brains.items():
                    results[metric][model][hemi] = dict()
                    #if hemi != 'left':
                    #    continue
                    all_res = numpy.zeros(shape=(len(hemi_data.keys()), 25, 15, 20))
                    fig, ax = pyplot.subplots(
                                              figsize=(20, 10),
                                              constrained_layout=True,
                                              )
                    areas = list()
                    for area, area_data in hemi_data.items():
                        '''
                        if area not in [
                                        #'IFG-full', 
                                        #'pSTS-full', 
                                        #'ATL-full', 
                                        #'language-network-full',
                                        ]:
                            continue
                        '''
                        areas.append(area)
                        for sub, all_sub_data in tqdm(area_data.items()):
                            sub_data = {k : numpy.nanmean(v, axis=0) for k, v in all_sub_data.items()}
                            for k, v in sub_data.items():
                                assert len(v.shape) == 2
                                assert v.shape[0] == 25
                                assert v.shape[1] > 100
                            ### matrix 
                            brain_mtrx = numpy.array([sub_data[k] for k in stimuli])
                            for t in range(25):
                                for train, test, iter_idx in zip(train_splits, test_splits, range(len(train_splits))):
                                    corrs = list()
                                    preds = list()
                                    for test_idx in test:
                                        real = brain_mtrx[test_idx, t, :]
                                        pred = numpy.nanmean([brain_mtrx[tr, t, :]*model_mtrx[tr, test_idx] for tr in train], axis=0)
                                        ### correlations
                                        if metric == 'spearman':
                                            avg_mult = numpy.nanmean([model_mtrx[tr, test_idx] for tr in train])
                                            norm_term = numpy.nanmean([brain_mtrx[tr, t, :]*avg_mult for tr in train], axis=0)
                                            corr = scipy.stats.spearmanr(real, pred-norm_term).statistic
                                            corrs.append(corr)
                                        elif metric == 'pairwise':
                                            preds.append((test_idx, pred))
                                    ### correlations
                                    if metric == 'spearman':
                                        all_res[areas.index(area), t, sub-1, iter_idx] = numpy.nanmean(corrs)
                                    ### pairwise
                                    elif metric == 'pairwise':
                                        accs = list()
                                        for test_one, pred_one in preds:
                                            for test_two, pred_two in preds:
                                                if test.index(test_two) <= test.index(test_one):
                                                    continue
                                                real_one = brain_mtrx[test_one, t, :]
                                                real_two = brain_mtrx[test_two, t, :]
                                                # we use the 'single pairwise metric' of
                                                # Beinborn, L., Abnar, S., & Choenni, R. (2019, April). 
                                                #  'Robust evaluation of language–brain encoding experiments.' 
                                                ### first pair
                                                correct = scipy.stats.spearmanr(pred_one, real_one).statistic
                                                wrong = scipy.stats.spearmanr(pred_one, real_two).statistic
                                                if correct > wrong:
                                                    acc = 1
                                                else:
                                                    acc = 0
                                                accs.append(acc)
                                                ### second pair
                                                correct = scipy.stats.spearmanr(pred_two, real_two).statistic
                                                wrong = scipy.stats.spearmanr(pred_two, real_one).statistic
                                                if correct > wrong:
                                                    acc = 1
                                                else:
                                                    acc = 0
                                                accs.append(acc)
                                        all_res[areas.index(area), t, sub-1, iter_idx] = numpy.nanmean(accs)
                        results[metric][model][hemi][area] = all_res[areas.index(area)]
                    for a_i, a in enumerate(areas):
                        ax.plot(
                                range(-5, 20),
                                numpy.nanmean(all_res[a_i], axis=(1, 2)),
                                label=a,
                                color=colors[a_i],
                                linewidth=7,
                                #res,
                                )
                        ax.fill_between(
                                range(-5, 20),
                                numpy.nanmean(all_res[a_i], axis=(1, 2))-scipy.stats.sem(all_res[a_i], axis=(1, 2)),
                                numpy.nanmean(all_res[a_i], axis=(1, 2))+scipy.stats.sem(all_res[a_i], axis=(1, 2)),
                                alpha=0.05,
                                color=colors[a_i],
                                )

                    if metric == 'spearman':
                        ax.plot(
                                numpy.array(range(-5, 20)),
                                bold_hrf,
                                label='BOLD HRF',
                                color='black',
                                linestyle='dashdot',
                                )
                        ax.set_ylim(bottom=-0.035, top=0.06)
                        ax.vlines(
                                  x=0.,
                                  ymin=-.035,
                                  ymax=.06,
                                  color='black',
                                  )
                        ax.hlines(
                                  y=0.,
                                  xmin=-5,
                                  xmax=19,
                                  color='black',
                                  )
                        ax.hlines(
                                  y=[-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05],
                                  xmin=-5,
                                  xmax=19,
                                  color='gray',
                                  linestyle='dashed',
                                  alpha=0.5
                                  )
                    elif metric == 'pairwise':
                        ax.plot(
                                numpy.array(range(-5, 20)),
                                bold_hrf+.5,
                                label='BOLD HRF',
                                color='black',
                                linestyle='dashdot',
                                )
                        ax.set_ylim(bottom=0.4, top=0.6)
                        ax.hlines(
                                  y=0.5,
                                  xmin=-5,
                                  xmax=19,
                                  color='black',
                                  )
                        ax.vlines(
                                  x=0.,
                                  ymin=.4,
                                  ymax=.6,
                                  color='black',
                                  )
                        ax.hlines(
                                  y=[.4, .42, .44, .46, .48, .52, .54, .56, .58, .6],
                                  xmin=-5,
                                  xmax=19,
                                  color='gray',
                                  linestyle='dashed',
                                  alpha=0.5
                                  )
                    ax.legend(
                              ncols=4,
                              fontsize=20,
                              )
                    pyplot.xticks(
                                  fontsize=20,
                                )
                    pyplot.yticks(
                                  fontsize=20,
                                )
                    pyplot.savefig(os.path.join(out, '{}_{}_{}.jpg'.format(hemi, case, model)),
                                   dpi=300
                                   )
                    counter.update(1)

out_f = os.path.join('pkls', 'rois', 'rsa_encoding')
os.makedirs(out_f, exist_ok=True)
with open(os.path.join(out_f, 'results.pkl'), 'wb'):
    pickle.dump(results, o)
