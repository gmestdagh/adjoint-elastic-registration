"""Classes to manage boundary conditions and a basic linear elastic model."""

import numpy as np
from scipy.sparse.linalg import factorized
from scipy.sparse import load_npz
import meshio


###############################################################################
# BOUNDARY CONDITIONS
###############################################################################
class NoBoundaryCondition:
    """Absence of boundary condition."""

    def __init__(self):
        pass

    def add_penalization_to_matrix(self, k):
        pass


class DirichletBoundaryCondition(NoBoundaryCondition):
    """A class to handle Dirichlet Boundary conditions."""

    def __init__(self, vertices: np.array):
        """Initialize Dirichlet indices."""
        super(DirichletBoundaryCondition, self).__init__()
        self.vertices = vertices
        self.indices = np.array(
            [3 * v + np.arange(3) for v in vertices]
        ).reshape(-1)

    def add_penalization_to_matrix(self, k):
        """Add large coefficients in matrix to enforce Dirichlet BC."""
        k[self.indices, self.indices] = np.finfo(k.dtype).max


###############################################################################
# LINEAR ELASTIC MODEL
###############################################################################
class LinearElasticModel:
    """Linear elastic model with precomputed stiffness matrix.

    This model loads a stiffness matrix from a file.
    The matrix should have saved with the function scipy.sparse.save_npz and
    should have the same size as the mesh position vector (three times the
    number of mesh vertices).

    To create a custom LinearElasticModel using your favorite FEM package,
    you may want to replace the line

          self.stiffness_matrix = load_npz(matrix_filename).tocsc()

    by a procedure which computes the stiffness matrix.
    """

    def __init__(self, mesh_filename: str, matrix_filename: str,
                 bc=NoBoundaryCondition()):
        """Load a stiffness matrix from a file and apply boundary conditions.

        :param mesh_filename:
        :param matrix_filename: Path of a .npz file containing the stiffness
            matrix in a Scipy format
        :param bc: Boundary condition
        """
        self.initial_position = meshio.read(mesh_filename).points.reshape(-1)

        self.stiffness_matrix = load_npz(matrix_filename).tocsc()
        bc.add_penalization_to_matrix(self.stiffness_matrix)
        self.solve_system = factorized(self.stiffness_matrix)

    def solve_adjoint(self, position: np.ndarray, rhs: np.ndarray):
        """Solve the adjoint system."""
        return self.solve_system(rhs)

    def solve_direct(self, forces: np.ndarray):
        """Compute displacement and return positions."""
        sol = self.solve_system(forces)
        return sol + self.initial_position
