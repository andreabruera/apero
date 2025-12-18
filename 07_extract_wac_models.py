import numpy
import os
import pickle
import scipy
import sklearn

from scipy import stats
from sklearn import metrics

from utils import read_stimuli

stimuli = read_stimuli()
ws = set()
for s in stimuli:
    verb = s.split()[0]
    if "'" in s:
        noun = s.split("'")[-1]
    else:
        noun = s.split()[-1]
    ws.add(verb)
    ws.add(noun)
ws = sorted(ws)

basic_folder = '../../counts/it/wac/'

#basic_folder = '../../counts/it/opensubs/'
#pos_f = os.path.join(basic_folder, 'it_opensubs_uncased_word_pos.pkl')
pos_f = os.path.join(basic_folder, 'it_wac_uncased_word_pos.pkl')
basic_folder = '../../counts/it/cc100/'
vocab_f = os.path.join(basic_folder, 'it_cc100_uncased_vocab_min_10.pkl')
pos = pickle.load(open(pos_f, 'rb'))
orig_chosen_ws = [w for w, p in pos.items() if sorted(p.items(), key=lambda item: item[1], reverse=True)[0][0] in ['NOUN', 'VERB', 'ADJ', 'ADV'] and len(w)>3 and '.' not in w and w not in ['noun', 'punct', 'misc', 'verb', 'propn', 'pron',]]
#freqs_f = os.path.join(basic_folder, 'it_wac_uncased_word_freqs.pkl')
#freqs_f = os.path.join(basic_folder, 'it_opensubs_uncased_word_freqs.pkl')
freqs_f = os.path.join(basic_folder, 'it_cc100_uncased_word_freqs.pkl')
freqs = pickle.load(open(freqs_f, 'rb'))
chosen_ws = list()
for w in orig_chosen_ws:
    try:
        freqs[w]
        chosen_ws.append(w)
    except KeyError:
        continue
top10000 = [w[0] for w in sorted([(k, freqs[k]) for k in chosen_ws], key=lambda item : item[1], reverse=True)][:10000]
top5000 = top10000[:5000]
top1000 = top10000[:1000]
top500 = top1000[:500]
top100 = top500[:100]
print(top500)
#basic_folder = '../../counts/it/wac/'
#vocab_f = os.path.join(basic_folder, 'it_wac_uncased_vocab_min_10.pkl')
vocab_f = os.path.join(basic_folder, 'it_cc100_uncased_vocab_min_10.pkl')
vocab = pickle.load(open(vocab_f, 'rb'))
#coocs_f = os.path.join(basic_folder, 'it_wac_coocs_uncased_min_10_win_20.pkl')
coocs_f = os.path.join(basic_folder, 'it_cc100_coocs_uncased_min_10_win_20.pkl')
coocs = pickle.load(open(coocs_f, 'rb'))
#basic_folder = '../../counts/it/wac/'
basic_folder = '../../counts/it/wac/'
old20_f = os.path.join(basic_folder, 'it_wac_10_min-uncased_OLD20.pkl')
old20 = pickle.load(open(old20_f, 'rb'))
### co-occurrences
w_coocs = numpy.zeros(shape=(len(ws), len(ws)))
for x, w in enumerate(ws):
    for y, w_two in enumerate(ws):
        try:
            cooc = coocs[vocab[w]][vocab[w_two]]
        except KeyError:
            cooc = 0.
        w_coocs[x, y] = cooc
phrase_coocs = numpy.zeros(shape=(36, len(ws)))
for s_i, s in enumerate(stimuli):
    verb = s.split()[0]
    if "'" in s:
        noun = s.split("'")[-1]
    else:
        noun = s.split(' ')[-1]
    vec = numpy.average([w_coocs[ws.index(verb)], w_coocs[ws.index(noun)]], axis=0)
    phrase_coocs[s_i] = vec
phrase_mtrx = sklearn.metrics.pairwise.cosine_similarity(phrase_coocs)
assert phrase_mtrx.shape == (36, 36)
### co-occurrences
'''
coocs5000 = numpy.zeros(shape=(len(ws), 5000))
for x, w in enumerate(ws):
    #for y, w_two in enumerate(ws):
    for y, w_two in enumerate(top5000):
        try:
            cooc = coocs[vocab[w]][vocab[w_two]]
        except KeyError:
            cooc = 0.
        coocs5000[x, y] = cooc
phrase_coocs = numpy.zeros(shape=(36, 5000))
for s_i, s in enumerate(stimuli):
    verb = s.split()[0]
    if "'" in s:
        noun = s.split("'")[-1]
    else:
        noun = s.split(' ')[-1]
    vec = numpy.average([coocs5000[ws.index(verb)], coocs5000[ws.index(noun)]], axis=0)
    phrase_coocs[s_i] = vec
mtrx5000 = sklearn.metrics.pairwise.cosine_similarity(phrase_coocs)
assert mtrx5000.shape == (36, 36)
'''
coocs10000 = numpy.zeros(shape=(len(ws), 10000))
for x, w in enumerate(ws):
    #for y, w_two in enumerate(ws):
    for y, w_two in enumerate(top10000):
        try:
            cooc = coocs[vocab[w]][vocab[w_two]]
        except KeyError:
            cooc = 0.
        coocs10000[x, y] = cooc
phrase_coocs = numpy.zeros(shape=(36, 10000))
for s_i, s in enumerate(stimuli):
    verb = s.split()[0]
    if "'" in s:
        noun = s.split("'")[-1]
    else:
        noun = s.split(' ')[-1]
    vec = numpy.average([coocs10000[ws.index(verb)], coocs10000[ws.index(noun)]], axis=0)
    phrase_coocs[s_i] = vec
mtrx10000 = sklearn.metrics.pairwise.cosine_similarity(phrase_coocs)
assert mtrx10000.shape == (36, 36)
### other values
surpr = numpy.zeros(shape=(36, 36))
ph_freqs = numpy.zeros(shape=(36, 36))
ph_old20 = numpy.zeros(shape=(36, 36))

for x, s_one in enumerate(stimuli):
    verb = s.split()[0]
    if "'" in s:
        noun = s.split("'")[-1]
    else:
        noun = s.split(' ')[-1]
    freq_x = (freqs[verb], freqs[noun])
    old20_x = (old20[verb], old20[noun])
    try:
        surpr_x = coocs[vocab[verb]][vocab[noun]]
    except KeyError:
        surpr_x = 0.
    for y, s_two in enumerate(stimuli):
        verb = s_two.split()[0]
        if "'" in s_two:
            noun = s_two.split("'")[-1]
        else:
            noun = s_two.split(' ')[-1]
        freq_y = (freqs[verb], freqs[noun])
        freq_sim = -scipy.spatial.distance.euclidean(freq_x, freq_y)
        ph_freqs[x, y] = freq_sim
        ph_freqs[y, x] = freq_sim
        old20_y = (old20[verb], old20[noun])
        old20_sim = -scipy.spatial.distance.euclidean(old20_x, old20_y)
        ph_old20[x, y] = old20_sim
        ph_old20[y, x] = old20_sim
        try:
            surpr_y = coocs[vocab[verb]][vocab[noun]]
        except KeyError:
            surpr_y = 0.
        surpr_sim = -abs(surpr_x-surpr_y)
        surpr[x, y] = surpr_sim
        surpr[y, x] = surpr_sim

vectors_f = os.path.join('models', 'vectors')
os.makedirs(vectors_f, exist_ok=True)
matrix_f = os.path.join('models', 'first_level_mtrx', 'euclidean')
os.makedirs(matrix_f, exist_ok=True)

for model, out_mtrx in [
                    #('surprisal-wac', surpr),
                    ('surprisal-cc100', surpr),
                    #('coocs-wac', phrase_mtrx),
                    #('coocs5000-wac', mtrx5000),
                    #('coocs10000-wac', mtrx10000),
                    ('coocs10000-cc100', mtrx10000),
                    #('freqs-wac', ph_freqs),
                    ('freqs-cc100', ph_freqs),
                    #('old20-wac', ph_old20),
                    ]:
    #ranks_vecs = scipy.stats.rankdata(mtrx, axis=1)
    #out_mtrx = numpy.corrcoef(ranks_vecs)

    ### vectors
    if 'coocs' in model:
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
