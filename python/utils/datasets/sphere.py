"""Load sphere dataset."""

from os.path import join, dirname
import numpy as np
from .datasets_utils import Dataset

DATA_FOLDER = join(dirname(__file__), '../../../data/sphere')
RESULTS_FOLDER = join(dirname(__file__), '../../../results/sphere')


def load(mesh_format='msh'):
    """Load sphere test dataset."""

    # Load mesh
    mesh_filename = join(DATA_FOLDER, f'sphere.{mesh_format}')

    # Load Dirichlet indices
    dirichlet_filename = join(DATA_FOLDER, 'dirichlet-vertices.npy')
    dirichlet_vertices = np.load(dirichlet_filename)

    matching_filename = join(DATA_FOLDER, 'matching-triangles.npy')
    matching_triangles = np.load(matching_filename)

    forces_vertices_filename = join(DATA_FOLDER, 'forces-vertices.npy')
    forces_vertices = np.load(forces_vertices_filename)

    pointcloud_filename = join(DATA_FOLDER, 'pointcloud.txt')
    pointcloud = np.loadtxt(pointcloud_filename, dtype=float)

    stiffness_filename = join(DATA_FOLDER, 'stiffness_matrix.npz')

    return Dataset(mesh_filename, dirichlet_vertices, matching_triangles,
                   forces_vertices, pointcloud, stiffness_filename)
