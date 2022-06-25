"""Common routines in dataset manipulation."""


class Dataset:
    """A simple class to store datasets."""
    def __init__(self, mesh_filename, dirichlet_vertices, matching_triangles,
                 forces_vertices, pointcloud, stiffness_filename=None):
        self.mesh_filename = mesh_filename
        self.dirichlet_vertices = dirichlet_vertices
        self.matching_triangles = matching_triangles
        self.forces_vertices = forces_vertices
        self.pointcloud = pointcloud
        self.stiffness_filename = stiffness_filename
