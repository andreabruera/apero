import numpy
import os
import scipy

from scipy import stats

from utils import read_stimuli

stimuli = read_stimuli()
matrix_f = os.path.join('models', 'first_level_mtrx', 'euclidean')
os.makedirs(matrix_f, exist_ok=True)

for rating in [
               'concreteness', 
               'familiarity',
               'imageability',
               ]:
    vecs = dict()
    missings = list()
    with open(os.path.join('data', '{}_phrase_ratings.tsv'.format(rating))) as i:
        for l_i, l in enumerate(i):
            if l_i == 0:
                continue
            line = l.strip().split('\t')
            ph = line[0]
            vec = line[1:]
            missing = [l_i for l_i, l in enumerate(vec) if l=='nan']
            missings.extend(missing)
            vecs[ph] = numpy.nanmean(numpy.array(vec, dtype=numpy.float32))
    mtrx = numpy.zeros(shape=(36, 36))
    for x, s in enumerate(stimuli):
        for y, s_two in enumerate(stimuli):
            ### highest possible rating is 5
            sim = 5-abs(vecs[s]-vecs[s_two])
            mtrx[x, y] = sim
            mtrx[y, x] = sim
    ### matrix
    with open(os.path.join(matrix_f, '{}_phrase.tsv'.format(rating)), 'w') as o:
        for x in range(mtrx.shape[0]):
            for y in range(mtrx.shape[1]):
                o.write('{}\t'.format(mtrx[x, y]))
            o.write('\n')
