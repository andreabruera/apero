import numpy
import os
import scipy

from scipy import stats

from utils import read_stimuli

stimuli = read_stimuli(categories=True)

food = ['cena', 'aperitivo', 'pranzo', 'pasta', 'pizza', 'risotto', 'ricetta']

abs_con = numpy.zeros(shape=(36, 36))
food_nofood = numpy.zeros(shape=(36, 36))
compositions = numpy.zeros(shape=(36, 36))

for x, s_one in enumerate(stimuli):
    abs_con_x = s_one[1]
    compo_x = s_one[2]
    food_x = False
    for f in food:
        if f in s_one[0]:
            food_x = True
    for y, s_two in enumerate(stimuli):
        abs_con_y = s_two[1]
        compo_y = s_two[2]
        food_y = False
        for f in food:
            if f in s_two[0]:
                food_y = True
        ### abstract/concrete
        if abs_con_x == abs_con_y:
            abs_con[x, y] = 1.
            abs_con[y, x] = 1.
        ### compositionality
        if compo_x == compo_y:
            compositions[x, y] = 1.
            compositions[y, x] = 1.
        ### compositionality
        if food_x == food_y:
            food_nofood[x, y] = 1.
            food_nofood[y, x] = 1.

vectors_f = os.path.join('models', 'vectors')
os.makedirs(vectors_f, exist_ok=True)
matrix_f = os.path.join('models', 'first_level_mtrx', 'spearman')
os.makedirs(matrix_f, exist_ok=True)
for model, mtrx in [
                    ('abstract-concrete-categorical', abs_con),
                    ('food-nofood-categorical', food_nofood),
                    ('composition-type-categorical', compositions),
                    ]:
    ranks_vecs = scipy.stats.rankdata(mtrx, axis=1)
    out_mtrx = numpy.corrcoef(ranks_vecs)

    ### vectors
    with open(os.path.join(vectors_f, '{}.tsv'.format(model)), 'w') as o:
        o.write('phrase\tvector\n')
        for phrase_i, phrase in enumerate(stimuli):
            o.write('{}\t'.format(phrase))
            for dim in out_mtrx[phrase_i]:
                o.write('{}\t'.format(float(dim)))
            o.write('\n')
    ### matrix
    with open(os.path.join(matrix_f, '{}.tsv'.format(model)), 'w') as o:
        for x in range(out_mtrx.shape[0]):
            for y in range(out_mtrx.shape[1]):
                o.write('{}\t'.format(out_mtrx[x, y]))
            o.write('\n')
