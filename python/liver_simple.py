"""Simple registration problem involving a liver."""
import os
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from utils.optimization_problems import AdjointControlProblem, ScipyWrapper
from utils.objective_functions import DataAttachment, L2Penalty
from utils.elasticity import DirichletBoundaryCondition, LinearElasticModel
from utils.datasets import liver
from utils.results_processing import plot_iterations, apply_positions

# Load dataset
# Available sequences are 's1, ..., s5'.
# Available meshes are 'remeshed' and 'default'.
ds = liver.load_sequence('s1', which_mesh='remeshed')

# Prepare output saving
save_folder = liver.RESULTS_FOLDER + '_simple'

###############################################################################
# CREATE PROBLEM
###############################################################################
# Initialize elastic problem
elastic_model = LinearElasticModel(
    mesh_filename=ds.mesh_filename,
    matrix_filename=ds.stiffness_filename,
    bc=DirichletBoundaryCondition(ds.dirichlet_vertices)
)

# Create optimization problem with penalty term
obj = DataAttachment(ds.matching_triangles, ds.pointcloud[0])
penalty = L2Penalty(1.0e-5, forces_vertices=ds.forces_vertices)
problem = AdjointControlProblem(obj, ds.forces_vertices, elastic_model,
                                penalty)

wrapper = ScipyWrapper(problem, save_folder=save_folder)

###############################################################################
# OPTIMIZATION PROCESS
###############################################################################
# Initialization
x0 = np.zeros(ds.forces_vertices.shape[0] * 3, dtype=np.float64)
wrapper.first_eval(x0)
gnorm0 = np.linalg.norm(wrapper.pb.g.reshape(-1), np.inf)

# Run Scipy solver (L-BFGS-B)
x_final, f_final, stats = fmin_l_bfgs_b(
    wrapper.fgobj,
    x0,
    m=200,
    factr=0.,
    pgtol=1e-5 * gnorm0,
    maxiter=1000,
    callback=wrapper.callback
)
print(stats['task'])

###############################################################################
# PROCESS RESULTS
###############################################################################
if save_folder is None:
    exit()

# Plot convergence curves
iterations_filename = os.path.join(save_folder, 'iterations.log')
eps_folder = os.path.join(save_folder, 'eps')
plot_iterations(iterations_filename, eps_folder)

# Create deformed meshes for viewing
apply_positions(save_folder, ds.mesh_filename)
