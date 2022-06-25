from os.path import join, isfile
import matplotlib.pyplot as plt
import numpy as np
import meshio


def plot_iterations(log_file: str, output_folder: str):
    """Plot convergence curves and saves them as eps figures."""

    # Read log file
    iters = np.loadtxt(log_file, dtype=int, skiprows=1, usecols=0)
    objval = np.loadtxt(log_file, dtype=float, skiprows=1, usecols=1)
    objgrad = np.loadtxt(log_file, dtype=float, skiprows=1, usecols=2)

    # Plot objective value
    plt.figure()
    plt.plot(iters, objval, '.-')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.xlabel("Iterations")
    plt.ylabel("Objective value")
    plt.title('Evolution of objective value')
    plt.savefig(join(output_folder, 'objective.eps'), transparent=False)

    # Plot gradient norm
    plt.figure()
    plt.plot(iters, objgrad, '.-')
    plt.yscale('log')
    plt.grid(True, which='both')
    plt.xlabel('Iterations')
    plt.ylabel('Gradient norm')
    plt.title('Evolution of optimality conditions')
    plt.savefig(join(output_folder, 'gradnorm.eps'), transparent=False)


def apply_positions(working_folder: str, reference_mesh_filename: str):
    """Create deformed meshes for vizualization"""

    # Prepare filenames and undeformed mesh
    pos_prefix = join(working_folder, 'npy', 'pos%03d.npy')
    frc_prefix = join(working_folder, 'npy', 'frc%03d.npy')
    mesh_prefix = join(working_folder, 'vtk', 'mesh%03d.vtk')

    mesh = meshio.read(reference_mesh_filename)
    mesh.cell_data = {}
    mesh.point_data = {}

    # Loop over files in the npy folder
    iter_idx = -1
    while True:
        iter_idx += 1
        pos_filename = pos_prefix % iter_idx
        frc_filename = frc_prefix % iter_idx
        mesh_filename = mesh_prefix % iter_idx

        if not isfile(pos_filename) or not isfile(frc_filename):
            break

        # Save position and forces into output mesh
        mesh.points = np.load(pos_filename).reshape(-1, 3)
        mesh.point_data = {'forces' : np.load(frc_filename).reshape(-1, 3)}
        mesh.write(mesh_filename)
