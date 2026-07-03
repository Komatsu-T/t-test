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
This simulation sets the population means of the two groups to be equal (a situation where the null hypothesis is true) and compares the rate at which Student's t-test and Welch's t-test falsely reject the null hypothesis—that is, the Type I error (α error). By varying the ratio of population variances and the ratio of sample sizes over a grid, it examines how far the α error of each test deviates from the nominal significance level under each condition.
For each condition, samples for the two groups are generated from normal distributions and both tests are applied; this is repeated 1,000,000 times to calculate the rejection rate (α error) and its confidence interval, the difference in α error between the two tests, and the quantiles of the p-values. The settings are managed in the [alpha_error_simulation] section of settings.toml, and the results are output in Parquet format.

Run the following commands on AWS EC2:

```bash
git clone https://github.com/Komatsu-T/t-test.git
cd t-test/welch-alpha-error-sim
bash setup.sh
bash run.sh
```
