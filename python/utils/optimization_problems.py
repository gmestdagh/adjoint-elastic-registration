"""Manage adjoint method and relationships with Scipy solvers."""
import os
from os.path import join
from time import time
import numpy as np
from numpy.linalg import norm
from scipy.optimize.optimize import MemoizeJac


class AdjointControlProblem:
    """An optimal control problem using an adjoint method.

    This class represents the problem

          min J(position) + R(forces),

    where J is a functional depending on position and R is a functional
    depending on forces.

    In this problem, the gradient in the space of forces is computed using an
    adjoint problem.
    """

    def __init__(self, position_objective, forces_vertices,
                 elastic_model, forces_objective=None):
        """Prepare problem data.

        :param position_objective: Objective function that depends on the
            mesh positions (i.e. which depends on the state)
        :param forces_vertices: Indices of vertices where forces apply
        :param elastic_model: Elastic model associated to the mesh
        :param forces_objective: Objective function the depends on the
            forces (i.e. which depends on the control)
        """

        self.pos_obj = position_objective
        self.forces_idx = \
            (3 * forces_vertices.reshape(-1, 1) + np.arange(3)).reshape(-1)
        self.elastic_model = elastic_model
        self.frc_obj = forces_objective

        # Initialize problem variables
        self.position = np.zeros_like(self.elastic_model.initial_position)
        self.forces = np.zeros_like(self.position)
        self.f = 0.
        self.fpos = 0.
        self.ffrc = 0.
        self.g = np.zeros(self.forces_idx.shape, dtype=float)
        self.f_eval = 0

    def fgobj(self, x: np.ndarray):
        """Evaluate the objective gradient and value for a control x.

        :param x: Vector of size (3 * n_forces_vertices) containing the
             value of nodal forces at the force vertices.
        """

        # Solve direct problem
        self.forces[self.forces_idx] = x
        self.position[:] = self.elastic_model.solve_direct(self.forces)

        # Evaluate position and optionnally force objective
        self.fpos, gpos = self.pos_obj.fgobj(self.position)
        self.f = self.fpos

        # Solve adjoint problem
        adjoint_state = self.elastic_model.solve_adjoint(self.position, gpos)
        self.g[:] = adjoint_state[self.forces_idx]

        if self.frc_obj is not None:
            self.ffrc, gfrc = self.frc_obj.fgobj(x)
            self.f += self.ffrc
            self.g += gfrc

        self.f_eval += 1
        return self.f, self.g

    def fobj(self, x: np.ndarray):
        """Evaluate the objective value for a control x.

        :param x: Vector of size (3 * n_forces_vertices) containing the
             value of nodal forces at the force vertices.
        """

        # Solve direct problem
        self.forces[self.forces_idx] = x
        self.position[:] = self.elastic_model.solve_direct(self.forces)

        # Evaluate position and optionnally force objective
        self.f = self.pos_obj.fobj(self.position)

        if self.frc_obj is not None:
            self.f += self.frc_obj.fobj(x)

        self.f_eval += 1
        return self.f

    # Logging utilities
    header_line = f"{'f':>13s}  {'gnorm':>13s}  {'|x|_inf':>13s}  " \
                  f"{'f_pos':>13s}  {'f_frc':>13s}  {'feval':>6s}"

    def get_log_line(self, x):
        """Return a formatted line containing current solver variables"""
        gnorm = norm(self.g)
        control_norm_inf = norm(x.reshape(-1, 3), axis=1).max()
        return f'{self.f:13.6e}  {gnorm:13.6e}  {control_norm_inf:13.6e}  ' \
               f'{self.fpos:13.6e}  {self.ffrc:13.6e}  {self.f_eval:6d}'


class ScipyWrapper:
    """A wrapper to use Sofa simulation in scipy.optimize.

    This wrapper provides a function fgObj to compute objective function and
    gradient, and a callback_function for plotting.
    """
    def __init__(self, pb, save_folder=None, verbose=True):
        """Initialize the wrapper.

        :param save_folder: folder where to save results
        :param verbose: whether to print iterations or not
        """

        self.pb = pb
        self.verbose = verbose

        # Files for logging
        self.save_iterations = (save_folder is not None)
        if self.save_iterations:
            for s in ['npy', 'vtk', 'eps']:
                os.makedirs(join(save_folder, s), exist_ok=True)
            self.logfile = join(save_folder, 'iterations.log')
            self.posfile = join(save_folder, 'npy', f'pos%03d.npy')
            self.frcfile = join(save_folder, 'npy', f'frc%03d.npy')

        # Solver state (for plotting)
        self.iter = -1
        self.start_time = 0.0

        # Add memoization tool
        self.memoize_jac = MemoizeJac(self.pb.fgobj)

    def first_eval(self, x0):
        """Problem initialization and first objective evaluation.

        :param x0: Initial control vector
        """
        self.pb.fgobj(x0)

        # Print iterations header
        line = f"{'iter':>5s}  " + self.pb.header_line + f"  {'time':>13s}"
        if self.verbose:
            print(line)
        if self.save_iterations:
            with open(self.logfile, 'w') as f:
                print(line, file=f)

        self.start_time = time()
        self.iter = -1
        self.callback(x0)

    def fgobj(self, x: np.ndarray):
        """Run simulation and measure objective function.

        The input x represents the force only for the force vertices.
        The gradient is returned with respect to the forces vertices only."""
        return self.pb.fgobj(x)

    def fobj(self, x: np.ndarray):
        """Evaluate only the objective value.

        This function provides an interface for solvers which require separate
        functions for value and gradient evaluations.
        """
        return self.memoize_jac(x)

    def gobj(self, x:np.array):
        """Evaluate only the objective gradient.

        This function provides an interface for solvers which require separate
        functions for value and gradient evaluations.
        """
        return self.memoize_jac.derivative(x)

    def callback(self, x: np.ndarray):
        """Function to execute at the end of a iteration, for printing, etc."""

        self.iter += 1
        current_time = time() - self.start_time
        line = f'{self.iter:5d}  ' + self.pb.get_log_line(x) \
               + f'  {current_time:13.3f}'

        # Print iteration in console
        if self.verbose:
            print(line)

        # Print iteration in log file
        if self.save_iterations:
            np.save(self.posfile % self.iter, self.pb.position)
            np.save(self.frcfile % self.iter, self.pb.forces)
            with open(self.logfile, 'a') as f:
                print(line, file=f)
