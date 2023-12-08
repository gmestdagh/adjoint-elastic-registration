"""Elastic models using Fenicsx.

To use this model, fenicsx should be installed.
"""
from dolfinx import fem
from dolfinx.io import XDMFFile
from dolfinx.fem import petsc
from petsc4py import PETSc
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


class NeoHookeanModel:
    """A Neo-Hookean model involving a Newton method."""
    def __init__(self, mesh_filename: str, young_modulus: float,
                 poisson_ratio: float, bc: DirichletBoundaryCondition,
                 verbose=True):

        # Load mesh and create vector space
        with XDMFFile(MPI.COMM_WORLD, mesh_filename, 'r') as xdmf:
            mesh = xdmf.read_mesh(name='Grid')
        fe_space = fem.functionspace(mesh,
                                     ('Lagrange', 1, (mesh.geometry.dim,)))

        # Save initial position
        self.dof_map = (
            np.array(mesh.geometry.input_global_indices, dtype=np.int32) \
                .reshape(-1, 1) * 3
            + np.arange(3, dtype=np.int32)
        ).reshape(-1)
        self.initial_position = np.zeros(mesh.geometry.x.size)
        self.initial_position[self.dof_map] = mesh.geometry.x.reshape(-1)

        # Define energy and its derivatives
        self.u = fem.Function(fe_space)
        w = ufl.TrialFunction(fe_space)
        v = ufl.TestFunction(fe_space)

        psi = self._nhk_energy(young_modulus, poisson_ratio, self.u) * ufl.dx
        self.energy_value = fem.form(psi)

        res = ufl.derivative(psi, self.u)
        self.energy_residual = fem.form(res)

        jac = ufl.derivative(res, self.u)
        self.energy_jacobian = fem.form(jac)

        # Create storage for Newton solver
        solution_storage = self.u.vector.copy()
        gradient_storage = petsc.create_vector(self.energy_residual)
        hessian_storage = petsc.assemble_matrix(self.energy_jacobian)
        hessian_storage.assemble()

        # Create bounds vectors to handle Drichlet conditions
        self.inverse_dof_map = np.zeros_like(self.dof_map, dtype=np.int32)
        self.inverse_dof_map[self.dof_map] = np.arange(self.dof_map.shape[0])
        constrained_dofs = self.inverse_dof_map[bc.indices]
        self.free_dofs = np.int32(np.where(np.logical_not(
            np.isin(np.arange(self.dof_map.shape[0]), constrained_dofs)
        ))[0])
        print(self.free_dofs.shape)

        lower_bounds = solution_storage.copy()
        lower_bounds.array[:] = PETSc.NINFINITY
        lower_bounds.array[constrained_dofs] = 0.

        upper_bounds = -lower_bounds

        # Create TAO solver
        self.tao = PETSc.TAO()
        self.tao.create()
        self.tao.setType('bnls')
        self.tao.setMaximumIterations(1000)
        self.tao.setInitial(solution_storage)
        self.tao.setObjectiveGradient(self._direct_objgrad, gradient_storage)
        self.tao.setHessian(self._direct_hessian, hessian_storage)
        self.tao.setVariableBounds(lower_bounds, upper_bounds)
        self.tao.setTolerances(gatol=1e-8, grtol=0., gttol=1e-8)
        self.tao.setFromOptions()

        ls = self.tao.getLineSearch()
        ls.setType('armijo')

        # Create storage for problem RHS (nodal forces)
        self.rhs = self.u.vector.copy()

    @staticmethod
    def _nhk_energy(young_modulus: float, poisson_ratio: float,
                    u: fem.Function):
        mu, lmbda = lame_parameters(young_modulus, poisson_ratio)
        d = len(u)

        I = ufl.variable(ufl.Identity(d))
        F = ufl.variable(I + ufl.grad(u))
        C = ufl.variable(F.T * F)
        Ic = ufl.variable(ufl.tr(C))
        J = ufl.variable(ufl.det(F))

        return (0.5 * mu) * (Ic - d) \
            - mu * ufl.ln(J) \
            + (0.5 * lmbda) * ufl.ln(J)**2

    @staticmethod
    def _linear_energy(young_modulus: float, poisson_ratio: float,
                  u: fem.Function):
        mu, lmbda = lame_parameters(young_modulus, poisson_ratio)

        d = len(u)
        epsilon = ufl.sym(ufl.grad(u))
        sigma = 2. * mu * epsilon \
            + lmbda * ufl.tr(epsilon) * ufl.Identity(d)
        return 0.5 * ufl.inner(epsilon, sigma)

    def _direct_objgrad(self, tao: PETSc.TAO, x: PETSc.Vec, g: PETSc.Vec):
        """Assemble energy gradient and return energy value."""
        x.copy(self.u.vector)

        # Check that point is feasible
        f = fem.assemble_scalar(self.energy_value)
        if np.isnan(f):
            f = PETSc.INFINITY
            #g.array[:] = 0.
            return f

        # Compute objective and assemble gradient
        f -= self.rhs.dot(x)
        g.array[:] = 0.
        petsc.assemble_vector(g, self.energy_residual)
        g.axpy(-1., self.rhs)

        nit = tao.getIterationNumber()
        print(f'{nit:3d}  {f:13.9e}  {np.linalg.norm(g[self.free_dofs]):13.9e}')
        return f

    def _direct_hessian(self, tao: PETSc.TAO, x: PETSc.Vec, H: PETSc.Mat,
                        P: PETSc.Mat):
        """Assemble energy hessian during direct system solving."""
        x.copy(self.u.vector)
        H *= 0
        petsc.assemble_matrix(H, self.energy_jacobian)
        H.assemble()
        return None

    def solve_direct(self, forces: np.ndarray):
        """Solve direct problem for a given force distribution.

        The TAO Newton solver is called to minimize the energy.
        """
        self.rhs[:] = forces[self.dof_map]
        self.rhs.assemble()
        self.tao.solve()
        return self.tao.solution[self.inverse_dof_map] + self.initial_position
