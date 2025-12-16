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

phrase_vecs = numpy.zeros(shape=(len(stimuli), 300))
verb_vecs = numpy.zeros(shape=(len(stimuli), 300))
noun_vecs = numpy.zeros(shape=(len(stimuli), 300))
for phrase_i, phrase in enumerate(stimuli):
    verb = phrase.split(' ')[0]
    noun = phrase.split("'")[-1] if "'" in phrase else phrase.split(' ')[-1]
    vec = numpy.average([ft[verb], ft[noun]], axis=0)
    assert vec.shape == (300,)
    phrase_vecs[phrase_i] = vec
    verb_vecs[phrase_i] = ft[verb]
    noun_vecs[phrase_i] = ft[noun]

##### spearman correlation
### transforming to ranks
#ranks_vecs = scipy.stats.rankdata(vecs, axis=1)
#mtrx = numpy.corrcoef(ranks_vecs)
##### cosine similarity
phrase_mtrx = sklearn.metrics.pairwise.cosine_similarity(phrase_vecs)
assert phrase_mtrx.shape == (36, 36)
verb_mtrx = sklearn.metrics.pairwise.cosine_similarity(verb_vecs)
noun_mtrx = sklearn.metrics.pairwise.cosine_similarity(noun_vecs)

vectors_f = os.path.join('models', 'vectors')
os.makedirs(vectors_f, exist_ok=True)
matrix_f = os.path.join('models', 'first_level_mtrx', 'cosine')
#matrix_f = os.path.join('models', 'first_level_mtrx', 'spearman')
os.makedirs(matrix_f, exist_ok=True)

for case, mtrx in [
                   ('verbnoun', phrase_mtrx),
                   ('verb', verb_mtrx),
                   ('noun', noun_mtrx),
                   ]:
    if case == 'verbnoun':
        ### vectors
        with open(os.path.join(vectors_f, 'fasttext_{}.tsv'.format(case)), 'w') as o:
            o.write('phrase\tvector\n')
            for phrase_i, phrase in enumerate(stimuli):
                o.write('{}\t'.format(phrase))
                for dim in phrase_vecs[phrase_i]:
                    o.write('{}\t'.format(float(dim)))
                o.write('\n')
    ### matrix
    with open(os.path.join(matrix_f, 'fasttext_{}.tsv'.format(case)), 'w') as o:
        for x in range(mtrx.shape[0]):
            for y in range(mtrx.shape[1]):
                o.write('{}\t'.format(mtrx[x, y]))
            o.write('\n')
