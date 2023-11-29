"""Example using SOFA to generate a stiffness matrix.

It will not work if SOFA is not installed in your system.
"""
import os
import numpy as np
from scipy.optimize import fmin_l_bfgs_b

from utils.optimization_problems import AdjointControlProblem, ScipyWrapper
from utils.objective_functions import DataAttachment, L2Penalty
from utils.elasticity import DirichletBoundaryCondition
from utils.fenicsx_models import LinearElasticModel
from utils.datasets import sphere
from utils.results_processing import plot_iterations, apply_positions

# Load dataset
ds = sphere.load(mesh_format='xdmf')

# Prepare output saving
save_folder = sphere.RESULTS_FOLDER + '_with_fenicsx'

###############################################################################
# CREATE PROBLEM
###############################################################################
# Load a SOFA elastic model
elastic_model = LinearElasticModel(
    mesh_filename=ds.mesh_filename,
    young_modulus=1.,
    poisson_ratio=0.49,
    bc=DirichletBoundaryCondition(ds.dirichlet_vertices)
)

# Create optimization problem with penalty term
obj = DataAttachment(ds.matching_triangles, ds.pointcloud)
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
