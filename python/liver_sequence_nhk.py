"""Estimate a sequence of local forces applied on the liver.

This script is used to generate the results in section 3.2 of our paper.
The average update time and number of iterations are displayed at the end of
the procedure.
"""
import os
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b
from time import time

from utils.optimization_problems import AdjointControlProblem, ScipyWrapper
from utils.objective_functions import DataAttachment
from utils.elasticity import DirichletBoundaryCondition
from utils.fenics_models import NeoHookeanModel
from utils.datasets import liver

# Load dataset
# Available sequences are s1, ..., s5.
# Available meshes are 'default' (used for data generation) and 'remeshed'.
seq_name = 's1'
ds = liver.load_sequence(seq_name, which_mesh='default', model='neohookean',
                         mesh_format='xdmf')

# Prepare output saving
save_folder = os.path.join(liver.RESULTS_FOLDER + '_sequences_nhk', seq_name)

###############################################################################
# CREATE PROBLEM
###############################################################################
# Initialize elastic problem
bc = DirichletBoundaryCondition(ds.dirichlet_vertices)
elastic_model = NeoHookeanModel(
    mesh_filename=ds.mesh_filename,
    young_modulus=4500.,
    poisson_ratio=0.49,
    bc=DirichletBoundaryCondition(ds.dirichlet_vertices),
    verbose=False
)

# Create optimization problem and solver
obj = DataAttachment(ds.matching_triangles, ds.pointcloud[0])
problem = AdjointControlProblem(obj, ds.forces_vertices, elastic_model)

wrapper = ScipyWrapper(problem, save_folder=None, verbose=False)

###############################################################################
# OPTIMIZATION PROCESS
###############################################################################
# Load ground truth forces
true_forces = np.load(
    os.path.join(liver.DATA_FOLDER, 'sequences', 'neohookean', seq_name,
                 'true-forces.npy')
)

# Initialize optimization
x0 = np.zeros(ds.forces_vertices.shape[0] * 3, dtype=np.float64)
wrapper.first_eval(x0)
gnorm0 = np.linalg.norm(wrapper.pb.g.reshape(-1), np.inf)

nruns = 50
times = np.zeros(nruns)
fcalls = np.zeros(nruns, dtype=int)
errs = np.zeros(nruns)

# Print header
print(f"  time  nit  fcalls  Error (%)  Termination message")

for i in range(nruns):
    # Update point cloud
    obj.update_point_cloud(ds.pointcloud[i])

    # Run optimization process
    start_time = time()
    x_final, f_final, stats = fmin_l_bfgs_b(
        wrapper.fgobj,
        x0,
        m=x0.shape[0],
        factr=10.,
        pgtol=5e-4 * gnorm0,
        maxiter=1000
    )
    times[i] = time() - start_time
    x0 = x_final

    # Evaluate estimation error
    estimated_force = x_final.reshape(-1, 3).sum(axis=0)
    diff = norm(estimated_force - true_forces[i]) / norm(true_forces[i])
    errs[i] = diff * 100
    fcalls[i] = stats["funcalls"]
    message = stats['task']
    nit = stats['nit']

    # Print log
    print(f'{times[i]:6.4f}  {nit:3d}  {fcalls[i]:6d}  {errs[i]:6.2f} %  '
          + message)

print('-----------------------------------------------------------------')
print(f'Average time : {times[1:].mean():.4f}')
print(f'Average fcalls : {fcalls[1:].mean():.2f}')
print(f'Average err : {errs[1:].mean():.3f}')
