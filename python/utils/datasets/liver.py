from os.path import join, dirname
import numpy as np
from .datasets_utils import Dataset

DATA_FOLDER = join(dirname(__file__), '../../../data/liver')
RESULTS_FOLDER = join(dirname(__file__), '../../../results/liver')


def load_sequence(seq_name: str, which_mesh='remeshed'):
    """Load a sequence of point clouds for successive estimations.

    :param seq_name: Index of the sequence
    :param which_mesh: mesh to load ['default', 'remeshed']
    """

    # Load dirichlet vertices and matching triangles from mesh folder
    mesh_folder = join(DATA_FOLDER, f'mesh-{which_mesh}')

    mesh_filename = join(mesh_folder, 'liver.msh')
    dirichlet_vertices = np.load(join(mesh_folder, 'dirichlet-vertices.npy'))
    matching_triangles = np.load(join(mesh_folder, 'matching-triangles.npy'))
    stiffness_filename = join(mesh_folder, 'stiffness_matrix.npz')

    # Load forces vertices and pointclouds from sequence folder
    seq_folder = join(DATA_FOLDER, 'sequences', seq_name)

    forces_vertices = np.load(
        join(seq_folder, f'forces-vertices-{which_mesh}.npy')
    )

    # Store all pointclouds in a big array
    pointcloud_prefix = join(seq_folder, 'pointclouds', 'pc%02d.txt')
    num_pointclouds = 50
    pointclouds = np.array(
        [np.loadtxt(pointcloud_prefix % j) for j in range(num_pointclouds)]
    )

    return Dataset(mesh_filename, dirichlet_vertices, matching_triangles,
                   forces_vertices, pointclouds, stiffness_filename)
