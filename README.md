# t-test
This repository contains simulation code for comparing the performance of Welch's t-test and Student's t-test. Its purpose is to quantitatively examine, through Monte Carlo simulation, how the difference in the assumptions underlying the two methods affects actual test results.
Multiple simulations are conducted individually under a range of objectives and conditions. The code for each simulation is organized and stored in its own subfolder.

## Repository Structure
The code for each simulation is stored in separate subfolders, organized by objective.

```
.
├── README.md
├── welch-alpha-error-sim/ 
├── simulation_02/
├── simulation_03/
└── ...
```
Each subfolder contains the complete set of code needed to run that simulation. For the detailed settings and procedures of each individual simulation, please refer to the "Simulation Details" section below.

## Execution Environment / Usage
The simulations in this repository were run on AWS EC2. The instance type used may differ from one simulation to another; details are provided in the "Simulation Details" section.
The runtime environment, including dependent libraries, is standardized with Docker, so that the simulations run in an identical software environment regardless of the instance used.
Since the specific execution steps vary from one simulation to another, please refer to the instructions for each individual simulation.

## Simulation Details

### welch-alpha-error-sim
