"""Elastic models using Fenics/Dolfin.

To use this module, Dolfin (*not* Dolfinx) should be installed.
"""

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import factorized
from scipy.optimize import Bounds, minimize

import dolfin as df

from .elasticity import DirichletBoundaryCondition


def lame_parameters(young_modulus: float, poisson_ratio: float):
    mu = 0.5 * young_modulus / (1. + poisson_ratio)
    lmbda = young_modulus * poisson_ratio / (1. + poisson_ratio) \
        / (1. - 2. * poisson_ratio)
    return mu, lmbda


class LinearElasticModel:
    """A Linear elastic model using fenics."""

    def __init__(self, mesh_filename: str, young_modulus: float,
                 poisson_ratio: float, bc: DirichletBoundaryCondition):
        """Create the stiffness matrix.

        The mesh file is supposed to be in format xdmf.
        """

        # Load mesh and create Vector space
        mesh = df.Mesh()
        with df.XDMFFile(mesh_filename) as f:
            f.read(mesh)
        self.initial_position = mesh.coordinates().reshape(-1)
        fe_space = df.VectorFunctionSpace(mesh, 'CG', 1)
        u = df.TrialFunction(fe_space)
        v = df.TestFunction(fe_space)

        # Assemble elasticity matrix
        mu, lmbda = lame_parameters(young_modulus, poisson_ratio)

        def epsilon(_u):
            return df.sym(df.nabla_grad(_u))

        a = (2. * mu * df.inner(epsilon(u), epsilon(v))
             + lmbda * df.div(u) * df.div(v)) * df.dx
        ai, aj, av = df.assemble(a).instance().mat().getValuesCSR()
        stiff_reorder = csr_matrix((av, aj, ai)).tocoo()

        # Deal with fenics automatic reordering
        dof_map = df.dof_to_vertex_map(fe_space)
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


def nhk_energy(young_modulus: float, poisson_ratio: float, u: df.Function):
    mu, lmbda = lame_parameters(young_modulus, poisson_ratio)

    d = u.geometric_dimension()
    identity_matrix = df.Identity(d)
    f_grad = identity_matrix + df.nabla_grad(u)
    cauchy_green = f_grad.T * f_grad
    jac_determinant = df.det(f_grad)

    return 0.5 * mu * (df.tr(cauchy_green) - d)\
        - mu * df.ln(jac_determinant) \
        + 0.5 * lmbda * df.ln(jac_determinant)**2


def energy_derivatives(psi, u: df.Function):
    fe_space = u.function_space()
    v = df.TestFunction(fe_space)
    du = df.TrialFunction(fe_space)

    energy = psi * df.dx
    residual = df.derivative(energy, u, v)
    jacobian = df.derivative(residual, u, du)

    return energy, residual, jacobian


class NeoHookeanModel(df.NonlinearProblem):
    """A Neo-Hookean elastic model involving a Newton method.

    The direct problem is solved using a trust-region Newton method
    available in Scipy.
    """
    def __init__(self, mesh_filename: str, young_modulus: float,
                 poisson_ratio: float, bc: DirichletBoundaryCondition,
                 verbose=True):
        """Create the model.

        The mesh should be in xdmf format.
        """

        # Initialise Dolfin problem for use with Newton method
        df.NonlinearProblem.__init__(self)
        self.newton_print = 2 if verbose else -1

        # Load mesh and create Vector space
        mesh = df.Mesh()
        with df.XDMFFile(mesh_filename) as f:
            f.read(mesh)
        self.initial_position = mesh.coordinates().reshape(-1)
        fe_space = df.VectorFunctionSpace(mesh, 'CG', 1)
        self.mesh = mesh
        self.vtd = df.vertex_to_dof_map(fe_space)
        self.dtv = df.dof_to_vertex_map(fe_space)

        # Prepare elastic model
        u = df.Function(fe_space)
        self.disp_petsc = u.vector()
        psi = nhk_energy(young_modulus, poisson_ratio, u)
        self._W, self._F, self._J = energy_derivatives(psi, u)

        # Prepare Dirichlet condition in scipy solver
        self.n = self.disp_petsc.size()
        self.dirichlet_indices = self.vtd[bc.indices]
        ub = np.full(self.n, np.inf)
        ub[self.dirichlet_indices] = 0.
        self.bounds = Bounds(-ub, ub)

        # Prepare rhs (a petsc vector)
        self.rhs = self.disp_petsc.copy()
        self.rhs.zero()

        # Prepare Newton solver
        self.newton_solver = df.NewtonSolver()
        prm = self.newton_solver.parameters
        prm['report'] = verbose
        prm['linear_solver'] = 'lu'
        prm['relative_tolerance'] = 1e-12
        prm['absolute_tolerance'] = 2e-14
        prm['error_on_nonconvergence'] = False
        prm['maximum_iterations'] = 20
        prm['lu_solver']['symmetric'] = True
        prm['lu_solver']['verbose'] = True

    # Functions for the elastic problem with scipy solver
    def fgobj(self, displacement: np.ndarray):
        """Objective value and gradient for Scipy solver."""
        self.disp_petsc[:] = displacement[:]
        objval = df.assemble(self._W) - self.disp_petsc.inner(self.rhs)
        if np.isnan(objval):
            return np.inf, np.zeros_like(displacement)
        grad = df.assemble(self._F)
        grad.axpy(-1, self.rhs)
        return objval, grad[:]

    def hess(self, displacement: np.ndarray):
        """Objective Hessian for Scipy solver."""
        self.disp_petsc[:] = displacement[:]
        h_petsc = df.assemble(self._J)
        ai, aj, av = h_petsc.instance().mat().getValuesCSR()
        h = csr_matrix((av, aj, ai))
        return h

    # Functions for the Petsc Newton solver
    def F(self, b, x):
        """Residual for Newton solver."""
        df.assemble(self._F, tensor=b)
        b.axpy(-1, self.rhs)
        b[self.dirichlet_indices] = 0.

    def J(self, a, x):
        """Residual Jaobian for Newton solver."""
        df.assemble(self._J, tensor=a)
        a.ident(self.dirichlet_indices)

    # Function for the inverse problem
    def solve_direct(self, forces: np.ndarray):
        """Solve direct problem for a given force distribution.

        The problem is solved as tight as possible using the trust-region
        method and the Newton solver finishes the job.
        """
        # Set rhs vector
        self.rhs[:] = forces[self.dtv]

        # Solve optimization problem
        x0 = np.asarray(self.disp_petsc)
        res = minimize(
            self.fgobj,
            x0,
            method='trust-constr',
            jac=True,
            hess=self.hess,
            tol=1.0e-12,
            bounds=self.bounds,
            options={
                'disp': True,
                'maxiter': 100000,
                'gtol': 1e-14,
                'xtol': 1e-10,
                'verbose': self.newton_print
            }
        )
        self.disp_petsc[:] = res['x']

        # Finish job with Newton solver
        self.newton_solver.solve(self, self.disp_petsc)

        return self.disp_petsc[self.vtd] + self.initial_position

    def solve_adjoint(self, position: np.ndarray, rhs: np.ndarray):
        """Assemble problem Hessian and solve adjoint problem"""
        self.disp_petsc[:] = (position - self.initial_position)[self.dtv]
        self.rhs[:] = rhs[self.dtv]

        h_petsc = df.assemble(self._J)
        h_petsc.ident(self.dirichlet_indices)
        adjoint_state = self.disp_petsc.copy()
        df.solve(h_petsc, adjoint_state, self.rhs)

        return adjoint_state[self.vtd]
