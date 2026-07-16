# t-test
This repository contains simulation code for comparing the performance of Welch's t-test and Student's t-test. Its purpose is to quantitatively examine, through Monte Carlo simulation, how the difference in the assumptions underlying the two methods affects actual test results.
Multiple simulations are conducted individually under a range of objectives and conditions. The code for each simulation is organized and stored in its own subfolder.

## Repository Structure
The code for each simulation is stored in separate subfolders, organized by objective.

```
.
├── README.md
├── welch-alpha-error-sim/ 
├── welch-alpha-error-sim-poisson/
├── sinh_arcsinh_moment_matching/
├── aaa/
├── bbb/
└── ...
```
Each subfolder contains the complete set of code needed to run that simulation. For the detailed settings and procedures of each individual simulation, please refer to the "Simulation Details" section below.

## Execution Environment / Usage
The simulations in this repository were run on AWS EC2 or local environments. Environments may differ from one simulation to another; details are provided in the "Simulation Details" section.
The runtime environment, including dependent libraries, is standardized with Docker, so that the simulations run in an identical software environment regardless of the instance used.
Since the specific execution steps vary from one simulation to another, please refer to the instructions for each individual simulation.

## Simulation Details

### welch-alpha-error-sim
This simulation sets the population means of the two groups to be equal (a situation where the null hypothesis is true) and compares the rate at which Student's t-test and Welch's t-test falsely reject the null hypothesis—that is, the Type I error (α error). By varying the ratio of population variances and the ratio of sample sizes over a grid, it examines how far the α error of each test deviates from the nominal significance level under each condition.
For each condition, samples for the two groups are generated from normal distributions and both tests are applied; this is repeated 1,000,000 times to calculate the rejection rate (α error) and its confidence interval, the difference in α error between the two tests, and the quantiles of the p-values. The settings are managed in the [alpha_error_simulation] section of settings.toml, and the results are output in Parquet format.

Run the following commands on AWS EC2. The generated results are stored in Amazon S3.
```bash
git clone https://github.com/Komatsu-T/t-test.git
cd t-test/welch-alpha-error-sim
bash setup.sh
tmux new -s sim
bash run.sh
```
The execution environment is as follows:

| Item | Details |
| --- | --- |
| Instance type | m7i.8xlarge  |
| OS / AMI | Ubuntu 24.04 LTS |
| Approx. runtime | about 10 minutes|

### welch-alpha-error-sim-poisson
This simulation is based on "welch-alpha-error-sim", with the distribution used to generate random numbers changed from the normal distribution to the Poisson distribution. Since the Poisson distribution has a mean and variance both equal to λ, assigning a common λ to both groups means that the equal-variance condition is always satisfied. Under this setup, the simulation examines how non-normality (the fact that normality does not hold) affects the Type I error (α error) of Student's t-test and Welch's t-test.

Run the following commands on AWS EC2. The generated results are stored in Amazon S3.
```bash
git clone https://github.com/Komatsu-T/t-test.git
cd t-test/welch-alpha-error-sim-poisson
bash setup.sh
tmux new -s sim
bash run.sh
```
The execution environment is as follows:

| Item | Details |
| --- | --- |
| Instance type | m7i.8xlarge  |
| OS / AMI | Ubuntu 24.04 LTS |
| Approx. runtime | about 25 minutes|

### sinh_arcsinh_moment_matching
A script that numerically solves for the sinh-arcsinh transformation parameters that produce a distribution with a target skewness and excess kurtosis. Given a grid of target values, it solves for the corresponding `(eps, delta)` at each point and writes the results to Parquet.
 
The normal distribution has its skewness and excess kurtosis fixed at 0; its only free parameters are the mean and the variance. Shaping a distribution itself requires transforming a normal variable. This script uses the sinh-arcsinh transformation of Jones & Pewsey (2009),

```
Y = sinh( (arcsinh(Z) + eps) / delta ),   Z ~ N(0, 1)
```
 
and solves for the `(eps, delta)` that realize the target (skewness, excess kurtosis) using `scipy.optimize.fsolve`.

Skewness and excess kurtosis cannot be specified independently. The constraints come in two tiers. First, for any distribution,
 
```
excess kurtosis >= skewness² - 2
```
 
holds. This bound does not depend on how the distribution is constructed: no distribution exists whose (skewness, excess kurtosis) pair violates it. On top of that, the range the sinh-arcsinh family can reach sits higher still.
 
| Skewness | 0.0 | 0.5 | 1.0 | 1.5 | 2.0 | 3.0 |
|---|---|---|---|---|---|---|
| sinh-arcsinh lower limit | -0.86 | -0.49 | 0.51 | 2.42 | 4.93 | 12.80 |
| Any distribution (skewness² - 2) | -2.00 | -1.75 | -1.00 | 0.25 | 2.00 | 7.00 |

Run the following commands on a local environment. The runtime is approximately 30 seconds.
```bash
git clone https://github.com/Komatsu-T/t-test.git
cd sinh_arcsinh_moment_matching
bash run.sh
```
