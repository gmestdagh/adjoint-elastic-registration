# Datasets

### The `sphere` dataset

It is an example of minimal dataset to use the procedure.
A minimal dataset should contain

* A mesh file (in format GMSH 2.2/ASCII if your want to use SOFA)

* The indices of vertices where Dirichlet conditions apply

* The indices of vertices where nodal forces apply

* The indices of the vertices of triangles that belong to the matching 
  surface (see the paper for details)

* A point cloud

* A stiffness matrix in Scipy format (`.npz`) saved with `scipy.sparse.save_npz`

Once your new dataset is ready, you can create a file similar to
`python/utils/datasets/sphere.py` to load it in the procedure.

### The `liver` dataset

This dataset is more complex.
It refers to Section 3.2 in the paper.

* `mesh-default` is the mesh that was used to generate the point clouds.
* `mesh-remeshed` is the mesh that was used to estimate the resultant force.
* `sequences` contains the five sequences `s1, ..., s5` of point clouds 
  mentioned in the paper, along with the true force used to generate point 
  clouds, the true position of the `default` mesh and the vertices on which 
  the true force has been applied to create a deformation.
  For each sequence, the vertices where reconstructed forces apply are also 
  specified.
