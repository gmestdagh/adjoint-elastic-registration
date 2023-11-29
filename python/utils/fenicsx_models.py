"""Elastic models using Fenicsx.

To use this model, fenicsx should be installed.
"""
from dolfinx import fem
from dolfinx.io import XDMFFile
from mpi4py import MPI

import ufl

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized

from .elasticity import DirichletBoundaryCondition


def lame_parameters(young_modulus: float, poisson_ratio: float):
    mu = 0.5 * young_modulus / (1. + poisson_ratio)
    lmbda = young_modulus * poisson_ratio / (1. + poisson_ratio) \
        / (1. - 2. * poisson_ratio)
    return mu, lmbda


class LinearElasticModel:
    """A linear elastic model using fenicsx."""

    def __init__(self, mesh_filename: str, young_modulus: float,
                 poisson_ratio: float, bc: DirichletBoundaryCondition):
        """Create the stiffness matrix.

        The mesh file is supposed to be in format xdmf.
        """

        # Load mesh and create vector space
        with XDMFFile(MPI.COMM_WORLD, mesh_filename, 'r') as xdmf:
            mesh = xdmf.read_mesh(name='Grid')
        fe_space = fem.functionspace(mesh,
                                     ('Lagrange', 1, (mesh.geometry.dim,)))
        u = ufl.TrialFunction(fe_space)
        v = ufl.TestFunction(fe_space)

        # Save initial position
        dof_map = (
            np.array(mesh.geometry.input_global_indices).reshape(-1, 1) * 3
            + np.arange(3)
        ).reshape(-1)
        self.initial_position = np.zeros(mesh.geometry.x.size)
        self.initial_position[dof_map] = mesh.geometry.x.reshape(-1)

        # Define linear problem and assemble stiffness matrix
        mu, lmbda = lame_parameters(young_modulus, poisson_ratio)

        def sigma(w):
            return 2. * mu * ufl.sym(ufl.grad(w)) \
                + lmbda * ufl.tr(ufl.sym(ufl.grad(w))) * ufl.Identity(len(w))
        
        a = fem.form(ufl.inner(sigma(u), ufl.grad(v)) * ufl.dx)
        stiff_reorder = fem.assemble_matrix(a).to_scipy().tocoo()

        # Deal with fenics automatic reordering
        stiff_reorder.col = dof_map[stiff_reorder.col]
        stiff_reorder.row = dof_map[stiff_reorder.row]
        self.stiffness_matrix = stiff_reorder.tocsc()

        bc.add_penalization_to_matrix(self.stiffness_matrix)
        self.solve_system = factorized(self.stiffness_matrix)

    def solve_adjoint(self, position: np.ndarray, rhs: np.ndarray):
        """Solve the adjoint system, return a nx3 vector and a flag.

        :param position Position where the adjoint system is evaluated
        :param rhs Right-hand side of the system (nx3 vector)
        """
        return self.solve_system(rhs)

    def solve_direct(self, forces: np.ndarray):
        """Compute displacement and return positions."""
        sol = self.solve_system(forces)
        return sol + self.initial_position
