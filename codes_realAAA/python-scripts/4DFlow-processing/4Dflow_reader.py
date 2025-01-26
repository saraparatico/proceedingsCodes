import glob
import scipy.io
import mat73
import numpy as np
from os.path import join
import os
import pyvista as pv
import pandas as pd
import SimpleITK as sitk

#---> root and files
root = '/media/rmunafo/TOSHIBA EXT1/PhD/adjoint/coding/AAA'
patients = os.listdir(join(root, 'data'))
for patient in patients:
    #---> paths
    segmentationPath = join(root, f'data/{patient}/analysis_results/segmentations')
    fourdpcPath = join(root, f'data/{patient}/matlab')

    #---> segmentation
    mat = scipy.io.loadmat(join(segmentationPath, 'abd_mask.mat'))
    np_mask = mat['Data']
    mask = sitk.GetImageFromArray(np.swapaxes(np_mask, 0, 2))
    dir = join(root, f'images/{patient}')
    if not os.path.exists(dir):
        os.mkdir(dir)
    sitk.WriteImage(mask, join(root, f'images/{patient}/abd_mask.seg.nrrd'))

    #---> magnitude
    #work around for scipy which cannot load matlab v7.3 files
    try:
        mat = mat73.loadmat(glob.glob(join(fourdpcPath, '4DPC_M_FFE.mat'))[0])
    except:
        mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '4DPC_M_FFE.mat'))[0])
    np_magnitude = mat['Data']
    img_magnitude = sitk.GetImageFromArray(np.swapaxes(np_magnitude, 0, 2))
    dir = join(root, f'images/{patient}')
    if not os.path.exists(dir):
        os.mkdir(dir)
    sitk.WriteImage(img_magnitude, join(root, f'images/{patient}/4DPC_M_FFE.seq.nrrd'))

    #---> velocity
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_AP.mat'))[0])
    np_velocityX = mat['Data']
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_FH.mat'))[0])
    np_velocityY = mat['Data'] * -1 #the y component looks inverted
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_RL.mat'))[0])
    np_velocityZ = mat['Data']
    np_velocityM = np.linalg.norm(np.stack((np_velocityX, np_velocityY, np_velocityZ), axis=0), axis=0)

    #---> get and save MetaData
    fieldnames = ['SpacingX', 'SpacingY', 'SpacingZ', 'Period', 'dt', 'Venc']
    df_arr = np.empty((1, len(fieldnames)))
    spacingX = df_arr[0, 0] = mat['MetaData'][0][0][0][0]
    spacingY = df_arr[0, 1] = mat['MetaData'][0][0][0][1]
    spacingZ = df_arr[0, 2] = mat['MetaData'][0][0][0][2]
    df_arr[0, 3] = mat['MetaData'][0][0][2][-1]
    df_arr[0, 4] = mat['MetaData'][0][0][2][1]
    df_arr[0, 5] = np.linalg.norm(mat['MetaData'][0][0][3])

    df = pd.DataFrame(df_arr, columns=fieldnames)
    df.to_csv(join(root, f'info/{patient}.csv'), index=False)

    #---> uniform grid
    volume = pv.wrap(np_mask)
    volume.spacing = (spacingX, spacingY, spacingZ)
    volume.rename_array('values', 'mask')
    mesh = volume.threshold(0.5, 'mask')

    #---> surface mesh
    pd_mesh = volume.threshold(0.5, 'mask').extract_surface()
    pd_mesh.clear_point_data()

    #---> save models (vtp and stl)
    pd_mesh.save(join(root, f'models/{patient}.vtp'))
    pd_mesh.save(join(root, f'models/{patient}.stl'))

    #---> read veocity time instants
    TF = mat['MetaData'][0][0][2]

    for time_index in range(len(TF)):
        np_velocity = np.stack((np_velocityX[:, :, :, time_index].flatten(order='F'),
                                np_velocityY[:, :, :, time_index].flatten(order='F'),
                                np_velocityZ[:, :, :, time_index].flatten(order='F')), axis=1)

        volume.point_data['velocity'] = np_velocity
        #---> save velocities
        meshInside = mesh.sample(volume)    #clip velocity data on mask
        dir = join(root, f'velocities/{patient}')
        if not os.path.exists(dir):
            os.mkdir(dir)
        meshInside.save(join(root, 'velocities/{}/velocity_{:02d}.vtu'.format(patient, time_index)))
