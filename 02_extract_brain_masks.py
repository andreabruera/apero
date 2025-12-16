import matplotlib
import nilearn
import numpy
import os
import scipy

from matplotlib import pyplot
from matplotlib.colors import LinearSegmentedColormap
from nilearn import datasets, plotting, surface
from scipy import stats

def plot_img(msk, area, original, out, fsaverage):
     for side in ['right', 'left']:
         work_msk = msk.copy()
         if side == 'right':
             work_msk[:int(work_msk.shape[0]*0.5), :, :] = 0.
         elif side == 'left':
             work_msk[int(work_msk.shape[0]*0.5):, :, :] = 0.
         if 'central' in area:
             portions = ['full', 'upper', 'lower',]
         else:
             portions = ['full']
         for p in portions:
             p_msk = work_msk.copy()
             if p == 'upper':
                 p_msk[:, :, :int(work_msk.shape[2]*0.66)] = 0.
             elif p == 'lower':
                 p_msk[:, :, int(work_msk.shape[2]*0.66):] = 0.
             exp_img = nilearn.image.load_img('../../neuroscience/dot_lunch_fast/derivatives/sub-01/ses-mri/func/sub-01_ses-mri_task-dotlunchfast_run-01_bold.nii')
             mask_img = nilearn.image.new_img_like(maps, p_msk)
             #### save masks
             mask_img.to_filename(os.path.join(out, '{}_{}-{}_mask.nii'.format(side, area, p)))

             exp_img = nilearn.image.index_img(exp_img, 0)
             atl_img = nilearn.image.resample_to_img(mask_img, exp_img, interpolation='nearest')

             print(area)
             print(sum(atl_img.get_fdata().flatten()))
             nilearn.plotting.plot_glass_brain(
                                           atl_img, 
                                           output_file=os.path.join(out, '{}_{}-{}_glass.jpg'.format(side, area, p),)
                                           )
             ### Right
             if side == 'right':
                 texture = surface.vol_to_surf(atl_img, fsaverage.pial_right,
                                               interpolation='nearest_most_frequent',
                         )
                 r = plotting.plot_surf_stat_map(
                             fsaverage.pial_right,
                             texture,
                             hemi='right',
                             title='{} - right hemisphere'.format(area),
                             threshold=0.05,
                             #bg_map=None,
                             cmap=cmap,
                             alpha=0.2,
                             )
                 r.savefig(os.path.join(out, \
                             'right_{}-{}.jpg'.format(area, p),
                             ),
                             dpi=600
                             )
                 pyplot.clf()
                 pyplot.close()
             ### Left
             if side == 'left': 
                 texture = surface.vol_to_surf(atl_img,
                                               fsaverage.pial_left,
                                               interpolation='nearest_most_frequent',
                         )
                 l = plotting.plot_surf_stat_map(
                             fsaverage.pial_left,
                             texture,
                             hemi='left',
                             title='{} - left hemisphere'.format(area),
                             colorbar=True,
                             threshold=0.05,
                             #bg_map=None,
                             cmap=cmap,
                             alpha=0.2,
                             )
                 l.savefig(os.path.join(out, \
                            'left_{}-{}.jpg'.format(area, p),
                             ),
                             dpi=600
                             )
                 pyplot.clf()
                 pyplot.close()

fsaverage = nilearn.datasets.fetch_surf_fsaverage()
dataset = nilearn.datasets.fetch_atlas_harvard_oxford('cort-maxprob-thr25-2mm')
maps = dataset['maps']
maps_data = maps.get_fdata()
labels = dataset['labels']

mapper = {
          'insula' : [
                      'Insular Cortex',
                      ],
          'precentral-gyrus' : [
                      'Precentral Gyrus',
                      ],
          'postcentral-gyrus' : [
                      'Postcentral Gyrus',
                      ],
          'SMA' : [
              'Juxtapositional Lobule Cortex (formerly Supplementary Motor Cortex)'
                      ],
          'vision-areas' : [
#'Lateral Occipital Cortex, superior division', 
'Lateral Occipital Cortex, inferior division', 
'Intracalcarine Cortex', 
'Lingual Gyrus', 
'Occipital Fusiform Gyrus', 
'Supracalcarine Cortex', 
'Occipital Pole'
                      ],
          }

color = 'white'
colormaps = {
          'insula' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'precentral-gyrus' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'postcentral-gyrus' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'SMA' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'vision-areas' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
            }

out = os.path.join('rois', 'masks')
os.makedirs(out, exist_ok=True)

for area, cmap in colormaps.items():
     relevant_labels = [i for i, l in enumerate(labels) if l in mapper[area]]
     msk = numpy.array([1. if v in relevant_labels else 0. for v in maps_data.flatten()]).reshape(maps_data.shape)
     plot_img(msk, area, maps, out, fsaverage)

### language network

mapper = {
          'IFG' : [
                   1, 2, 7, 8 
                   ],
          'ATL' : [
                   4, 10,
                   ],
          'pSTS' : [
                    5, 6, 11, 12, 
                    ],
          'language-network' : [
                    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 
                    ],
          }
color = 'white'
colormaps = {
          'IFG' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'ATL' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'pSTS' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
          'language-network' : LinearSegmentedColormap.from_list(
                                            "mycmap",
                                           [
                                            color,
                                            'paleturquoise',
                                                ]),
            }


mask_img = nilearn.image.load_img(os.path.join('rois', 'allParcels-language-SN220.nii'))
for area, cmap in colormaps.items():
    msk = mask_img.get_fdata().copy()
    for voxel_val in mapper[area]:
         ### one-ing the relevant area
         msk[msk==voxel_val] = 21.
    ### zeroing everything else than the area
    msk[msk!=21.] = 0
    msk[msk==21.] = 1.
    plot_img(msk, area, mask_img, out, fsaverage)
