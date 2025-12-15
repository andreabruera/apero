import os

folder = os.path.join(
                      '/',
                      'data',
                      'tu_bruera',
                      'neuroscience',
                      'dot_lunch_fast',
                      'derivatives',
                      )

vps = set()

for root, direc, fz in os.walk(folder):
    for f in fz:
        if 'tsv' not in f:
            continue
        with open(os.path.join(root, f)) as i:
            lines = i.readlines()
        header = [val.strip() for val in lines[0].split('\t')]
        for l_i, l in enumerate(lines[1:]):
            split_l = [val.strip() for val in l.split('\t')]
            ### we keep only verbs or nouns
            if 'abs' in split_l[-1] or 'con' in split_l[-1]:
                w = split_l[-2]
            else:
                continue
            ### noun
            if ' ' in w or "'" in w:
                continue
            ### verb
            noun_l = [val.strip() for val in lines[l_i+3].split('\t')]
            assert 'con' in noun_l[-1] or 'abs' in noun_l[-1]
            vp = w + ' ' + noun_l[-2]
            vps.add((vp, split_l[-1]))

assert len(vps) == 36

out_f = 'data'
os.makedirs(out_f, exist_ok=True)

with open(os.path.join(out_f, 'trials.tsv'), 'w') as o:
    o.write('phrase\tconcreteness\ttrial_type\n')
    for vp, cats in sorted(vps, key=lambda item : item[1]):
        abs_con = cats[:3]
        abs_con = 'concrete' if abs_con=='con' else 'abstract'
        other = cats[3:]
        if other == 'Coer':
            other += 'cion'
        o.write('{}\t{}\t{}\n'.format(vp, abs_con, other.lower()))
