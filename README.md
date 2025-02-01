# proceedingsCodes
Codes associated to the chapter "Estimation of optimal inlet boundary conditions for blood flow assessment in abdominal aortic aneurysm using variational data assimilation", which was submitted to the Proceedings of the FEniCS Conference 2024.

## Requirements & installation
This project was developed using Python 3.8.10. It is recommended to run the scripts in a Python environment compatible with this version. The development was carried out using PyCharm as the Integrated Development Environment (IDE).

For the installation of FEniCS, please refer to the official installation guide at: https://fenicsproject.org/download/archive/

## Description
This repository contains two main directories, each corresponding to a different simulation setup:

- **"codes_syntheticGeometries"**: Includes all the necessary code to run simulations on simple synthetic geometries, such as a 2D rectangular domain and a 3D cylindrical domain.
- **"codes_realAAA"**: Designed to run simulations on patient-specific abdominal aortic aneurysm (AAA) geometries.

### Common Structure
Both directories share a similar structure, containing the following key files:

- **"main.py"**: The core script of the project. Execute this file to run the simulations in both cases.
- **"v0.json"**: Stores all simulation parameters. To modify the working conditions (e.g., switch between 2D and 3D simulations), update this file. No other code modifications are required.
- **"solver.py"**: Contains the numerical solvers used for the simulations. This file is not meant to be executed directly.
- **"problem.py"**: Defines the problem class, handling mesh reading, function space definition, and the inclusion of control input and observation functions. These functions are generated using `utils.py`. Neither `problem.py` nor `utils.py` should be executed directly.

### Key Differences Between Directories
Despite the structural similarities, the two directories differ in their input geometric and physical data:

- **"codes_syntheticGeometries"**:  
  - Contains an `"initialization-files"` folder, which includes scripts for generating the initial meshes.  
  - These scripts must be executed **before** running the simulation to generate the required input mesh.  
  - Boundary conditions and observations are synthetically generated via functions in `utils.py`, which are automatically invoked after setting up `v0.json` and executing `main.py`.

- **"codes_realAAA"**:  
  - Contains an `"init-data"` folder, which includes scripts for converting meshes and processing input data:  
    - `convertMesh.py`: Converts input `.vtu` and `.vtp` files into the required mesh format.  
    - `inletW.py` and `obsW.py`: Read and process inlet and observation data for the simulations.

## Workflow
The following sequence of steps must be followed for both directories:

1. **Set up your problem**: Properly configure the `v0.json` file.
2. **Create/Convert the Mesh**: Use the scripts in the relevant subfolder to create or convert the mesh into the required format.
3. **Generate Boundary Conditions and Observations**: Prepare the necessary boundary conditions and observations.
4. **Run the Main Simulation**: Once the mesh and boundary conditions are set, execute `main.py` to run the simulation.

### Detailed Workflows

#### Workflow for "codes_syntheticGeometries":
1. **Set up your problem**: Properly configure the `v0.json` file.
2. **Create the Mesh**: Use the scripts in the `"initialization-files"` subfolder to generate the mesh in the required format.
3. **Run the Main Simulation**: Once the mesh and boundary conditions (defined in `v0.json`) are set, execute `main.py` to run the simulation.  
   - In this case, boundary conditions and observations are generated automatically, as `main.py` calls the appropriate functions from `problem.py` and `utils.py`.

#### Workflow for "codes_realAAA":
1. **Set up your problem**: Properly configure the `v0.json` file.
2. **Convert the Mesh**: Use `init-data/convertMesh.py` to read and convert the mesh into the required format.
3. **Generate Boundary Conditions and Observations**:  
   - Use `init-data/inletW.py` and `init-data/obsW.py` to read and convert velocity data into the required format.  
   - If using the IPCS scheme, execute an additional step with `project-inlet.py` and `project-obs.py` to project the generated data onto the appropriate function space.  
   - **Note:** `components.py` is another script related to observation generation, specifically for working with noisy data.
4. **Run the Main Simulation**: Once the mesh and boundary conditions are set, execute `main.py` to run the simulation.

## Usage
To run the simulation, modify the "v0.json" file to set the desired parameters (e.g., geometry, simulation conditions). Then, execute "main.py". The structure is designed to allow easy adjustments for different geometries and simulation setups.

To work with the simulations, I decided to use a virtual environment. I created it using PyCharm, but then I often worked from the terminal, following these steps:
- Creation of a Virtual Environment using Pycharm
- Once the virtual environment is created, you can activate it by navigating to the folder containing the environment and running:
(on bash) "source /path/to/folder/bin/activate"
- To deactivate the virtual environment, simply run:
(on bash) "deactivate"
- Inside the virtual environment, I typically used MPI (Message Passing Interface) to launch simulations, as follows:
(on bash) "mpirun -n 12 python3 main.py --v 0".
This command utilizes 12 processors on a workstation equipped with 24 CPU cores and 64 GB of RAM.

However, this was my preferred setup for running simulations and you are free to choose the method that best suits your needs, whether you prefer to use fewer processors or run the simulation without MPI support.
