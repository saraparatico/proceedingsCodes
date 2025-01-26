import pyvista as pv
import scipy.io
import os
from os.path import join
import glob
import numpy as np

#---> root and files
root = '/media/rmunafo/TOSHIBA EXT/PhD/adjoint/data/AAA'
patients = os.listdir(join(root, 'data'))
for patient in patients:
    # ---> paths
    imagePath = join(root, f'images/{patient}/cut')
    fourdpcPath = join(root, f'data/{patient}/matlab')

    # ---> velocity
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_AP.mat'))[0])
    np_velocityX = mat['Data']
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_FH.mat'))[0])
    np_velocityY = mat['Data'] * -1 #the y component looks inverted
    mat = scipy.io.loadmat(glob.glob(join(fourdpcPath, '*BC4O_RL.mat'))[0])
    np_velocityZ = mat['Data']
    np_velocityM = np.linalg.norm(np.stack((np_velocityX, np_velocityY, np_velocityZ), axis=0), axis=0)

    #---> Get spacing
    spacingX = mat['MetaData'][0][0][0][0] * 1000
    spacingY = mat['MetaData'][0][0][0][1] * 1000
    spacingZ = mat['MetaData'][0][0][0][2] * 1000

    #---> uniform grid
    volume = pv.read(join(imagePath, f'abd_mask.seg.nrrd')) #clip mask
    volume.spacing = (spacingX, spacingY, spacingZ)
    volume.rename_array('ImageScalars', 'mask')
    mesh = volume.threshold(0.5, 'mask')

    #---> surface mesh
    pd_mesh = volume.threshold(0.5, 'mask').extract_surface()
    pd_mesh.clear_point_data()

    #---> save models (vtp and stl)
    pd_mesh.save(join(root, f'models/cut/{patient}.vtp'))
    pd_mesh.save(join(root, f'models/cut/{patient}.stl'))

    #---> read veocity time instants
    TF = mat['MetaData'][0][0][2]

    for time_index in range(len(TF)):
        np_velocity = np.stack((np_velocityX[:, :, :, time_index].flatten(order='F') * 1000,
                                np_velocityY[:, :, :, time_index].flatten(order='F') * 1000,
                                np_velocityZ[:, :, :, time_index].flatten(order='F') * 1000), axis=1)

        volume.point_data['velocity'] = np_velocity
        # ---> save velocities
        meshInside = mesh.sample(volume)  # clip velocity data on mask
        dir = join(root, f'velocities/{patient}/cut')
        if not os.path.exists(dir):
            os.mkdir(dir)
        meshInside.save(join(root, 'velocities/{}/cut/velocity_{:02d}.vtu'.format(patient, time_index)))