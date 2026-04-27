# GEOM2057 Lectures 6 and 7

This repository contains the Julia teaching scripts, lecture PDFs, and supporting assets for the second part of the GEOM2057 course, centred on Lectures 6 and 7.

## Topics Covered in the Second Part

Lecture 6 focuses on deterministic inversion and optimisation:

- Problem setup for a non-linear inverse problem.
- Synthetic data generation.
- Newton method inversion.
- Gauss-Newton inversion.
- A harder non-linear comparison case.
- Local versus global search behaviour.
- Particle Swarm Optimisation (PSO).
- Very Fast Simulated Annealing (VFSA).

Lecture 7 focuses on uncertainty and Bayesian inversion:

- Probability review and probability density functions.
- Joint, marginal, and conditional probability.
- Bayes' rule and posterior probability.
- Grid-search Bayesian inversion.
- Monte Carlo sampling.
- Markov chains.
- Metropolis-Hastings MCMC.
- Reading traces, marginals, and predictive uncertainty.
- Geophysical inversion examples.
- Implicit neural representations and future directions.

## Repository Contents

Lecture 6 includes:

- `L6/01-problem-setup-derivatives-etc.jl`: shared definitions and derivatives for the baseline non-linear inverse problem.
- `L6/02-data-generation.jl`: synthetic data generation for the baseline inverse problem.
- `L6/03-inversion-NewtonMethod.jl`: a Newton-method inversion example.
- `L6/04-inversion-GaussNewton.jl`: a Gauss-Newton inversion example.
- `L6/05-problem-setup-hard.jl`: shared setup for the harder, more non-linear comparison case.
- `L6/06-data-generation-hard.jl`: shared noisy synthetic data for the hard comparison case.
- `L6/07-inversion-GaussNewton-hard.jl`: local Gauss-Newton on the hard comparison case.
- `L6/08-inversion-PSO-hard.jl`: Particle Swarm Optimisation on the hard comparison case.
- `L6/09-inversion-VFSA-hard.jl`: Very Fast Simulated Annealing on the hard comparison case.

Lecture 7 includes:

- `L7/01-probability-review.jl`: introductory probability and PDF concepts for the lecture.
- `L7/02-bayesian-grid-search.jl`: grid-based Bayesian inversion demonstration.
- `L7/03-markov-chain-demo.jl`: simple Markov chain demonstration.
- `L7/04-metropolis-hastings-geophysical.jl`: MCMC demonstration on a geophysical inverse problem.
- `L7/05-inr-geophysical-inversion.jl`: high-level implicit neural representation inversion example.
- `L7/L7.pdf`: Lecture 7 slides.
- `L7/Figures/`: generated lecture figures and supporting images.

## Getting Started with Julia

The Julia scripts in Lecture 6 only use Julia standard libraries such as `LinearAlgebra`, `Printf`, and `Random`, so you do not need to install extra Julia packages to run the `.jl` files. Lecture 7 examples may additionally rely on plotting packages already set up in the teaching environment.

### 1. Install Julia

Download and install Julia from the official site:

https://julialang.org/downloads/

After installation, confirm Julia is available from a terminal:

```powershell
julia --version
```

If that command is not found, restart VS Code or add Julia to your system `PATH`.

### 2. Open the lecture code folder

From VS Code or a terminal, work inside:

```text
GEOM2057
```


### 3. Run the Julia scripts

For Lecture 6, move into `GEOM2057/L6` and run the scripts below.

Run the baseline problem setup:

```powershell
julia .\01-problem-setup-derivatives-etc.jl
```

Run the Newton inversion example:

```powershell
julia .\03-inversion-NewtonMethod.jl
```

Run the Gauss-Newton inversion example:

```powershell
julia .\04-inversion-GaussNewton.jl
```

Run the shared hard-problem setup:

```powershell
julia .\05-problem-setup-hard.jl
```

Run the shared noisy-data generation example:

```powershell
julia .\06-data-generation-hard.jl
```

Run the hard-case Gauss-Newton comparison:

```powershell
julia .\07-inversion-GaussNewton-hard.jl
```

Run the hard-case PSO global-optimisation example:

```powershell
julia .\08-inversion-PSO-hard.jl
```

Run the hard-case VFSA global-optimisation example:

```powershell
julia .\09-inversion-VFSA-hard.jl
```

For Lecture 7, move into `GEOM2057/L7` and run the lecture demos or compile the slides as needed.

Example:

```powershell
julia .\04-metropolis-hastings-geophysical.jl
```

Each script prints the main intermediate or final results in the terminal.

## Suggested Workflow for Students

1. Run `01-problem-setup-derivatives-etc.jl` to understand the toy inverse problem setup.
2. Run `03-inversion-NewtonMethod.jl` and inspect the printed iteration history.
3. Run `04-inversion-GaussNewton.jl` and compare it against Newton's method.
4. Run `05-problem-setup-hard.jl` and `06-data-generation-hard.jl` to inspect the shared hard case and the noisy observations used by all three comparison methods.
5. Run `07-inversion-GaussNewton-hard.jl`, `08-inversion-PSO-hard.jl`, and `09-inversion-VFSA-hard.jl` to compare local and global search on the same hard inverse problem.
6. Move to `L7` to review the probability, Bayesian inversion, Markov chain, MCMC, and INR examples that build the uncertainty-analysis part of the course.
