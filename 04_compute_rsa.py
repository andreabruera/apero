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

font_setup('../../fonts')


models = {
         'surprisal-cc100' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'surprisal-cc100.tsv'),
         'surprisal-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'surprisal-wac.tsv'),
         'coocs10000-cc100' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs10000-cc100.tsv'),
         'coocs10000-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs10000-wac.tsv'),
         'surprisal-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'surprisal-wac.tsv'),
         'coocs-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'coocs-wac.tsv'),
         'old20-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'old20-wac.tsv'),
         'freqs-wac' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'freqs-wac.tsv'),
         'concreteness-continuous' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'concreteness_phrase.tsv'),
         'phrase-length' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'phrase-levenshtein_phrase.tsv'),
         'phrase-levenshtein' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'phrase-length_phrase.tsv'),
         'familiarity' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'familiarity_phrase.tsv'),
         'imageability' : os.path.join('models', 'first_level_mtrx', 'euclidean', 'imageability_phrase.tsv'),
         'fasttext-phrase' : os.path.join('models', 'first_level_mtrx', 'cosine', 'fasttext_verbnoun.tsv'),
         'fasttext-verb' : os.path.join('models', 'first_level_mtrx', 'cosine', 'fasttext_verb.tsv'),
         'fasttext-noun' : os.path.join('models', 'first_level_mtrx', 'cosine', 'fasttext_noun.tsv'),
         'conceptnet' : os.path.join('models', 'first_level_mtrx', 'cosine', 'conceptnet_verbnoun.tsv'),
         'concreteness-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'abstract-concrete-categorical.tsv'),
         'food-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'food-nofood-categorical.tsv'),
         'composition-type-categorical' : os.path.join('models', 'first_level_mtrx', 'spearman', 'composition-type-categorical.tsv'),
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

for model, model_path in models.items():
    print(model)

    ft_tri = load_model(model_path)

    with tqdm() as counter:
        for case in (
                     'rois', 
                     'rois_selected-features',
                     ):
            out = os.path.join('plots', 'time_resolved', 'rsa', case)
            os.makedirs(out, exist_ok=True)

            with open(os.path.join('brains', 'first_order_mtrx', '{}.pkl'.format(case)), 'rb') as i:
                brains = pickle.load(i)
            for hemi, hemi_data in brains.items():
                all_res = numpy.zeros(shape=(len(hemi_data.keys()), 25, 15))
                fig, ax = pyplot.subplots(
                                          figsize=(20, 10),
                                          constrained_layout=True,
                                          )
                areas = list()
                for area, area_data in hemi_data.items():
                    areas.append(area)
                    '''
                    res = numpy.zeros(shape=(20, ))
                    for t in range(20):
                        ts = list()
                        for sub, sub_data in area_data.items():
                            assert sub_data.shape == (20, 36, 36)
                            t_tri = [sub_data[t, x, y] for x in range(36) for y in range(36) if y>x]
                            ts.append(t_tri)
                        t_tri = numpy.average(ts, axis=0)
                        corr = scipy.stats.spearmanr(ft_tri, t_tri).statistic
                        res[t] = corr
                    '''
                    for sub, sub_data in area_data.items():
                        assert sub_data.shape == (25, 36, 36)
                        for t in range(25):
                            t_tri = [sub_data[t, x, y] for x in range(36) for y in range(36) if y>x]
                            '''
                            others = list()
                            for sub_other in range(1, 16):
                                if sub_other == sub:
                                    continue
                                other_tri = [area_data[sub_other][t, x, y] for x in range(36) for y in range(36) if y>x]
                                others.append(other_tri)
                            others_tri = numpy.average(others, axis=0)
                            corr = scipy.stats.spearmanr(others_tri, t_tri).statistic
                            '''
                            corr = scipy.stats.spearmanr(ft_tri, t_tri).statistic
                            all_res[areas.index(area), t, sub-1] = corr
                for a_i, a in enumerate(areas):
                    ax.plot(
                            range(-5, 20),
                            numpy.nanmean(all_res[a_i], axis=1),
                            label=a,
                            color=colors[a_i],
                            linewidth=7,
                            #res,
                            )
                    ax.fill_between(
                            range(-5, 20),
                            numpy.average(all_res[a_i], axis=1)-scipy.stats.sem(all_res[a_i], axis=1),
                            alpha=0.05,
                            color=colors[a_i],
                            )
                    ax.fill_between(
                            range(-5, 20),
                            numpy.average(all_res[a_i], axis=1)+scipy.stats.sem(all_res[a_i], axis=1),
                            alpha=0.05,
                            color=colors[a_i],
                            )
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
