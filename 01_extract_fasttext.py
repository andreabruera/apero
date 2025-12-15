import fasttext
import numpy
import os
import scipy
import sklearn

from scipy import stats
from sklearn import metrics

from utils import read_stimuli

stimuli = read_stimuli()

ft_path = '/data/u_bruera_software/word_vectors/it/cc.it.300.bin'
ft = fasttext.load_model(ft_path)

vecs = numpy.zeros(shape=(len(stimuli), 300))
for phrase_i, phrase in enumerate(stimuli):
    verb = phrase.split(' ')[0]
    noun = phrase.split("'")[-1] if "'" in phrase else phrase.split(' ')[-1]
    vec = numpy.average([ft[verb], ft[noun]], axis=0)
    assert vec.shape == (300,)
    vecs[phrase_i] = vec

##### spearman correlation
### transforming to ranks
#ranks_vecs = scipy.stats.rankdata(vecs, axis=1)
#mtrx = numpy.corrcoef(ranks_vecs)
##### cosine similarity
mtrx = sklearn.metrics.pairwise.cosine_similarity(vecs)
assert mtrx.shape == (36, 36)

vectors_f = os.path.join('models', 'vectors')
os.makedirs(vectors_f, exist_ok=True)
matrix_f = os.path.join('models', 'first_level_mtrx', 'cosine')
#matrix_f = os.path.join('models', 'first_level_mtrx', 'spearman')
os.makedirs(matrix_f, exist_ok=True)

### vectors
with open(os.path.join(vectors_f, 'fasttext_verbnoun.tsv'), 'w') as o:
    o.write('phrase\tvector\n')
    for phrase_i, phrase in enumerate(stimuli):
        o.write('{}\t'.format(phrase))
        for dim in vecs[phrase_i]:
            o.write('{}\t'.format(float(dim)))
        o.write('\n')
### matrix
with open(os.path.join(matrix_f, 'fasttext_verbnoun.tsv'), 'w') as o:
    for x in range(mtrx.shape[0]):
        for y in range(mtrx.shape[1]):
            o.write('{}\t'.format(mtrx[x, y]))
        o.write('\n')
