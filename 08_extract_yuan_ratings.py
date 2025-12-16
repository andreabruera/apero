import numpy
import os
import re
import scipy

from scipy import stats

from utils import read_stimuli

stimuli = read_stimuli()
ratings = {k : {'familiarity' : dict(), 'concreteness' : dict(), 'imageability' : dict()} for k in stimuli}
absent = set()
subs = dict()

folder = os.path.join('data', 'norming_data_yuan_tsv')
sub_ages = dict()
for f in os.listdir(folder):
    if 'SUBJECT_LIST' in f:
        with open(os.path.join(folder, f)) as i:
            for l in i:
                line = l.strip().split('\t')
                try:
                    assert len(line) in [5, 6]
                except AssertionError:
                    continue
                if len(line) == 6:
                    name_idx = 1
                    sex_idx = 3
                    age_idx = 4
                elif len(line) == 5:
                    try:
                        name_idx = 0
                        sex_idx = 2
                        age_idx = 3
                        age = float(line[age_idx])
                    except ValueError:
                        name_idx = 1
                        sex_idx = 3
                        age_idx = 4
                age = float(line[age_idx])
                sex = line[sex_idx]
                assert sex in ['M', 'F', '']
                if sex == '':
                    sex = 'other'
                name = line[name_idx]
                sub_ages[name] = (age, sex)

for f in os.listdir(folder):
    if 'tsv' not in f or 'SUBJECT_LIST' in f or 'subjList_verb' in f:
        continue
    sub = re.findall(r'[A-Z]+\d|[A-Z]+', f)
    assert len(sub) == 1
    sub = sub[0]

    with open(os.path.join(folder, f)) as i:
        for l_i, l in enumerate(i):
            line = l.split('\t')
            try:
                assert len(line) == 5
            except AssertionError:
                #print(line)
                pass
            marker = True
            if l_i == 0:
                try:
                    assert 'fam' in line[2]
                    assert 'conc' in line[3]
                    assert 'immag' in line[4]
                    continue
                except AssertionError:
                    #print(f)
                    #print(line)
                    marker = False
                    continue
            if marker == False:
                continue
            try:
                fam = float(line[2])
            except ValueError:
                fam = numpy.nan
            try:
                conc = float(line[3])
            except ValueError:
                conc = numpy.nan
            try:
                imag = float(line[4])
            except ValueError:
                imag = numpy.nan
            if line[1] not in stimuli:
                if line[1] in ['confezione il regalo', 'confizionare il regalo']:
                    ph = 'confezionare il regalo'
                elif line[1] in ['confezione il pacco', 'confizionare il pacco']:
                    ph = 'confezionare il pacco'
                elif line[1] in ["annullare l'apertivo", "annullare l'aperti vo"]:
                    ph = "annullare l'aperitivo"
                elif line[1] in ["organizzare l'apertivo"]:
                    ph = "organizzare l'aperitivo"
                else:
                    absent.add(line[1])
                    continue

            else:
                ph = line[1]
            for k in ratings.keys():
                for k_two in ratings[k].keys():
                    if sub not in ratings[k][k_two].keys():
                        ratings[k][k_two][sub] = list()
            if fam != numpy.nan:
                ratings[ph]['familiarity'][sub].append(fam)
            if conc != numpy.nan:
                ratings[ph]['concreteness'][sub].append(conc)
            if imag != numpy.nan:
                ratings[ph]['imageability'][sub].append(imag)
            if sub not in subs.keys():
                subs[sub] = 0
            subs[sub] += 1

avail = set([len(v) for _ in ratings.values() for __ in _.values() for v in __.values()])
for k, v in ratings.items():
    for k_two, v_two in v.items():
        for sub, sub_data in v_two.items():
            if len(sub_data) == 2:
                print((sub, sub_data))
### we retain only subjects with more than 30 evaluations
present_subs = [s for s, v in subs.items() if v>=30]

mtrxs = {k : numpy.zeros(shape=(36, len(present_subs))) for k in ['concreteness', 'imageability', 'familiarity']}

for ph, ph_data in ratings.items():
    for cat, cat_data in ph_data.items():
        for sub, sub_data in cat_data.items():
            if sub not in present_subs:
                continue
            try:
                val = sub_data[0]
            except IndexError:
                val = numpy.nan
            mtrxs[cat][stimuli.index(ph), present_subs.index(sub)] = val

for rating, mtrx in mtrxs.items():
    with open(os.path.join('data', '{}_phrase_ratings.tsv'.format(rating)), 'w') as o:
        o.write('phrase\t')
        for _ in range(len(present_subs)):
            sub = present_subs[_]
            try:
                age = int(sub_ages[sub][0])
            except KeyError:
                age = 'NA'
            try:
                sex = sub_ages[sub][1]
            except KeyError:
                sex= 'NA'
            o.write('{}_age-{}_sex-{}\t'.format(sub, age, sex))
        o.write('\n')
        for x in range(36):
            o.write('{}\t'.format(stimuli[x]))
            for y in range(len(present_subs)):
                o.write('{}\t'.format(float(mtrx[x, y])))
            o.write('\n')
