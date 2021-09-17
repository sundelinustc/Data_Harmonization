#!usr/bin/env python
#pip install neuroHarmonize
import pandas as pd
import numpy as np
#from neuroHarmonize import harmonizationLearn
from neuroHarmonize.harmonizationNIFTI import createMaskNIFTI # can't be in the same folder with previous versions of neuroHarmonize

# Making a mask image for training set
nifti_list = pd.read_csv('brain_images_paths.csv')
print('\nnifti_list=\n',nifti_list)
nifti_avg, nifti_mask, affine, hdr0 = createMaskNIFTI(nifti_list, threshold=0)

# Flattern the images to 2D np.array
from neuroHarmonize.harmonizationNIFTI import flattenNIFTIs
nifti_array = flattenNIFTIs(nifti_list, 'thresholded_mask.nii.gz')

# Model & adjusted data
import neuroHarmonize as nh
covars = pd.read_csv('covs.csv')
print('\ncovars=\n',covars)
my_model, nifti_array_adj = nh.harmonizationLearn(nifti_array, covars)
#my_model, nifti_array_adj = nh.harmonizationLearn(nifti_array, covars, smooth_terms=['Age'])
print('\nnifti_array_adj=\n',nifti_array_adj)
# np.savetxt("Data_harmonized.csv", nifti_array_adj, delimiter=",") # save adjusted data (not in nifti format)
#nh.saveHarmonizationModel(my_model, 'MY_MODEL')

# Adjusted images
from neuroHarmonize.harmonizationNIFTI import applyModelNIFTIs
# my_model = nh.loadHarmonizationModel('MY_MODEL') # load pre-trained model
applyModelNIFTIs(covars, my_model, nifti_list, 'thresholded_mask.nii.gz')