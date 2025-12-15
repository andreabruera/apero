import numpy
import os
import pickle
import random
import scipy

from matplotlib import pyplot

from utils import font_setup

font_setup('../../fonts')

out = os.path.join('plots', 'time_resolved')
os.makedirs(out, exist_ok=True)


ft = list()
#with open(os.path.join('models', 'first_level_mtrx', 'spearman', 'fasttext_verbnoun.tsv')) as i:
with open(os.path.join('models', 'first_level_mtrx', 'cosine', 'fasttext_verbnoun.tsv')) as i:
    for l in i:
        l = [float(v) for v in l.strip().split('\t')]
        ft.append(l)
ft = numpy.array(ft)
assert ft.shape == (36, 36)
#ft_tri = numpy.triu(ft, k=1)
ft_tri = [ft[x, y] for x in range(36) for y in range(36) if y>x]
#ft_tri = [(x, y) for x in range(36) for y in range(36) if y>x]

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
                'teal',
                'paleturquoise',
                'mediumturquoise',
                'lightskyblue',
                'paleturquoise',
                'teal',
                'lightsalmon',
               'darkgray',
                ]
colors = random.sample(colors, k=len(colors))

for case in ('rois', 'rois_selected-features'):
    with open(os.path.join('brains', 'first_order_mtrx', '{}.pkl'.format(case)), 'rb') as i:
        brains = pickle.load(i)
    for hemi, hemi_data in brains.items():
        all_res = numpy.zeros(shape=(len(hemi_data.keys()), 20, 15))
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
                assert sub_data.shape == (20, 36, 36)
                for t in range(20):
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
                    range(-5, 15),
                    numpy.average(all_res[a_i], axis=1),
                    label=a,
                    color=colors[a_i],
                    linewidth=7,
                    #res,
                    )
            ax.fill_between(
                    range(-5, 15),
                    numpy.average(all_res[a_i], axis=1)-scipy.stats.sem(all_res[a_i], axis=1),
                    alpha=0.05,
                    color=colors[a_i],
                    )
            ax.fill_between(
                    range(-5, 15),
                    numpy.average(all_res[a_i], axis=1)+scipy.stats.sem(all_res[a_i], axis=1),
                    alpha=0.05,
                    color=colors[a_i],
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
                  xmax=15,
                  color='black',
                  )
        ax.hlines(
                  y=[-0.03, -0.02, -0.01, 0.01, 0.02, 0.03, 0.04, 0.05],
                  xmin=-5,
                  xmax=15,
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
        pyplot.savefig(os.path.join(out, '{}_{}_fasttext.jpg'.format(hemi, case)),
                       dpi=300
                       )
