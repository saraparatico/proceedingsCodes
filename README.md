# proceedingsCodes
Codes associated to the chapter "Estimation of optimal inlet boundary conditions for blood flow assessment in abdominal aortic aneurysm using variational data assimilation", which was submitted to the Proceedings of the FEniCS Conference 2024.

## Requirements & installation
This project was developed using Python 3.8.10. It is recommended to run the scripts in a Python environment compatible with this version. The development was carried out using PyCharm as the Integrated Development Environment (IDE).

For the installation of FEniCS, please refer to the official installation guide at: https://fenicsproject.org/download/archive/

## Description
This repository contains two main directories, each corresponding to different simulation setups:
- "codes_syntheticGeometries": it includes all the necessary code to run simulations on simple synthetic geometries, such as a 2D rectangular domain and a 3D cylindrical domain.
- "codes_realAAA": it is designed to run simulations on patient-specific abdominal aortic aneurysm geometries.

Both directories have a similar structure, which includes the following key files:
- "main.py": it is the core of the project and is the file to execute in both cases to run the simulations.
- "v0.json": it contains all the simulation parameters. To adjust the working conditions (e.g., switching from 2D to 3D simulations), simply modify this file. All other code files remain unchanged.
- "solver.py": it contains the solvers used for the simulations. It's not supposed to be run itself.
- "problem.py": it file defines the problem class. It is responsible for reading the mesh, defining function spaces, and including the functions that read control inputs and observations. The control and observation functions are generated by functions defined in "utils.py". There two codes are not supposed to be run itself

On the other hand, there is a significant difference in the input geometric and physical data between the two directories:
- In "codes_syntheticGeometries", there is a folder named "initialization-files" which contains files for generating the initial meshes. There codes must be run before the simulation, in order to generate the mesh that must be put in input to the problem.
  Boundary conditions and observations are then synthetically generated using functions defined in "utils.py", which is authomaticaly involvd after setting uo the v0.json and run the file "main.py".
- In "codes_realAAA", there is a folder named "init-data". It includes files that allow you to convert the mesh ("convertMesh.py") into the required format from an input ".vtu" and ".vtp" file. It also includes files that read the inlet ("inletW.py") and observation ("obsW.py") conditions for the simulations.

## Workflow
For both directories, the following sequence of steps must be followed:
- Set up your problem by properly compiling the v0.json file.
- Create/Convert the Mesh: Use the scripts in the appropriate subfolder to create or convert the mesh into the desired format.
- Generate Boundary Conditions and Observations: The boundary conditions and observations must be generated next. 
- Run the Main Simulation: After the mesh and boundary conditions are set, run "main.py" to execute the simulation.

More in details..
### Workflow for "codes_syntheticGeometries":
- Set up your problem by properly compiling the v0.json file.
- Create the Mesh: Use the scripts in the appropriate subfolder ("initialization-files") to create the mesh into the desired format.
- Run the Main Simulation: After the mesh and boundary desired conditions are specified on v0.json, run "main.py" to execute the simulation. In this case, the generation of boundary conditions and observations is done authomaticately because "main.py" calls propriate functions contained in "problem.py" and "utils.py".

### Workflow for "codes_realAAA":
- Set up your problem by properly compiling the v0.json file.
- Convert the Mesh: Use the scripts in the appropriate subfolder ("init-data\convertMesh.py") to read and convert the mesh into the desired format.
- Generate Boundary Conditions and Observations: The boundary conditions and observations must be generated next, using the scripts in the appropriate subfolder ("init-data\inletW.py" and "init-data\obsW.py") to read and convert the velocity data into the desired format. Moreover, if you want to work with the IPCS scheme, you need to execute an addivite step using "project-inlet.py" and "project-obs.py" which project the data previously created onto the appropriate function space.
- Run the Main Simulation: After the mesh and boundary conditions are set, run "main.py" to execute the simulation.

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
