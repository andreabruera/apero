import numpy
import os
import scipy

from utils import read_stimuli

def levenshtein(seq1, seq2):
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = numpy.zeros((size_x, size_y))
    for x in range(size_x):
        matrix [x, 0] = x
    for y in range(size_y):
        matrix [0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix [x,y] = min(
                    matrix[x-1, y] + 1,
                    matrix[x-1, y-1],
                    matrix[x, y-1] + 1
                )
            else:
                matrix [x,y] = min(
                    matrix[x-1,y] + 1,
                    matrix[x-1,y-1] + 1,
                    matrix[x,y-1] + 1
                )
    #print (matrix)
    return (matrix[size_x - 1, size_y - 1])

stimuli = read_stimuli()

len_mtrx = numpy.zeros(shape=(36, 36))
leven_mtrx = numpy.zeros(shape=(36, 36))

for x, s in enumerate(stimuli):
    for y, s_two in enumerate(stimuli):
        len_sim = -abs(len(s)-len(s_two))
        len_mtrx[x, y] = len_sim
        len_mtrx[y, x] = len_sim
        leven_sim = -levenshtein(s, s_two)
        leven_mtrx[x, y] = leven_sim
        leven_mtrx[y, x] = leven_sim

matrix_f = os.path.join('models', 'first_level_mtrx', 'euclidean')
os.makedirs(matrix_f, exist_ok=True)

for rating, mtrx in [
                     ('phrase-length', len_mtrx),
                     ('phrase-levenshtein', leven_mtrx),
                     ]:
    ### matrix
    with open(os.path.join(matrix_f, '{}_phrase.tsv'.format(rating)), 'w') as o:
        for x in range(mtrx.shape[0]):
            for y in range(mtrx.shape[1]):
                o.write('{}\t'.format(mtrx[x, y]))
            o.write('\n')
