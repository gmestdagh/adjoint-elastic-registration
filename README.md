# adjoint-elastic-registration

This repository provides the code associated to the following paper
(preprint available [here](https://hal.inria.fr/hal-03691913)):

* Guillaume Mestdagh, Stéphane Cotin. 
  An Optimal Control Problem for Elastic Registration and Force Estimation in 
  Augmented Surgery. 
  *MICCAI 2022 - 25th International Conference on Medical Image Computing 
  and Computer Assisted Intervention*, Sep 2022, Singapore, Singapore.
  
The repository contains an implementation of an adjoint method and an 
optimization procedure to perform elastic registration from partial surface 
data.
Our procedures only involve linear elasticity.
Some simple examples are provided to understand how it works, along with
utilities to prepare results for vizualization 
(with [Paraview](https://www.paraview.org/) for instance).

### Run a simple example

You will need an installation of Python with the packages `numpy`, `scipy`, 
`matplotlib`, `meshio`, `trimesh`, `rtree` installed.

First, clone the repository, then run the first example:
```bash
$ git clone https://github.com/gmestdagh/elastic-organ-registration.git
$ cd elastic-organ-registration/python
$ python3 sphere_simple.py

 iter              f          gnorm        |x|_inf          f_pos          f_frc   feval           time
    0   5.760117e-02   3.071415e+00   0.000000e+00   5.760117e-02   0.000000e+00       1          0.000
    1   5.329404e-02   4.355900e-01   7.379684e-03   5.329404e-02   0.000000e+00       5          0.227
    2   1.349479e-02   5.363131e-01   8.152418e-03   1.349479e-02   0.000000e+00       9          0.427
  
  ...
  
  181   1.158547e-07   4.032264e-05   3.230549e-02   1.158547e-07   0.000000e+00     191          5.015
  182   1.150924e-07   1.962825e-05   3.226678e-02   1.150924e-07   0.000000e+00     192          5.044
CONVERGENCE: NORM_OF_PROJECTED_GRADIENT_<=_PGTOL
```
If the program has worked, you should see the iterations appear in the console.
The successive deformed meshes are saved in the folder `results/sphere/vtk` for
viewing with Paraview.

### Available examples

The example scripts are gathered in the `python` folder.
The first three examples use precomputed stiffness matrices so that no 
finite element package is necessary to run them.

* `sphere_simple.py`: A very basic example involving a truncated sphere mesh 
  and a mesh covering the whole deformed surface.
  Results are saved in the folder `results/sphere`.

* `liver_simple.py`: Single estimation of a very local force distribution
  involving a liver mesh. See [the paper](https://hal.inria.fr/hal-03691913), 
  Section 3.2, for more details about data generation.
  Results are saved in `results/liver_simple`.

* `liver_sequence.py`: Test case presented in section 3.2 of 
  [the paper](https://hal.inria.fr/hal-03691913). The resultant force is
  updated for a series of 50 successive pointclouds and compared to the ground
  truth. 
  No deformed is saved for this test case but the summary of successive 
  estimations is printed in the console.

* `liver_with_sofa.py`: Same as `liver_simple.py`, but the stiffness matrix 
  is computed using [SOFA](https://www.sofa-framework.org/) and it is possible
  to choose elastic parameters.
  To run this case, you will need to install the [SOFA](https://www.sofa-framework.org/)
  finite element framework (code works with version v20.12), along with
  [SofaPython3](https://github.com/sofa-framework/sofapython3) and the
  [Caribou](https://github.com/mimesis-inria/caribou) plugin.

* `liver_sequence_nhk.py`: Same as `liver_sequence.py`, but the dataset and the
  reconstruction both feature a Neo-Hookean model for deformations.
  To run this case, you should have the Dolfin package installed (the old one, 
  not dolfinx).
  Note that this case is very slow and takes several hours to complete.
  You should consider running it during the night. 
  Also, do not worry about the "Newton solver did not converge warning", the
  residual tolerance has been set very tight on purpose.

As the access to the Sparse Data Challenge dataset is restricted by the
organizers, we do not provide an example associated with this test case.

### Structure of the repository

* `python`: example scripts
* `python/utils`: implementation of the adjoint method and optimization 
  utilities
* `data`: datasets used in example scripts.

The `results` folder is created by example scripts when some results are saved.

### Compute stiffness matrix at execution time

Our datasets come with precomputed stiffness matrices.
However, to play with elastic parameters, it is more convenient to have a finite
element package compute the stiffness matrix during script execution.

For this reason, we provide a class in `python/utils/sofa_models.py` which
assembles the stiffness matrix using the [SOFA](https://www.sofa-framework.org/)
framework.
To use it, you will need to install
* [SOFA](https://www.sofa-framework.org/) (v20.12)
* [SofaPython3](https://github.com/sofa-framework/sofapython3). If you compile
  SOFA, it is possible to add SofaPython3 by checking
  `SOFA_FETCH_SOFAPYTHON3` in Cmake.
* [Caribou](https://github.com/mimesis-inria/caribou)

If you prefer to assemble the stiffness matrix using your favorite finite 
element package, you can implement your own class.
Have a look at the `LinearElasticModel` class from `python/utils/elasticity.py`
to see which functions are necessary.

### Add another dataset

To try the procedure with your own data, you can add a folder in the `data`
directory.
To see an example of minimal test case, have a look at the dataset in
`data/sphere` to see what is necessary.
Once your dataset folder is ready, you can implement a simple class to load it
following the example in `python/utils/datasets/sphere.py`.
