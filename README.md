# GEOM2057 Lecture 6

This repository keeps only the Julia teaching scripts and the README tracked for Lecture 6.

The repository currently includes:

- `L6/01-problem-setup-derivatives-etc.jl`: shared definitions and derivatives for the baseline non-linear inverse problem.
- `L6/02-data-generation.jl`: synthetic data generation for the baseline inverse problem.
- `L6/03-inversion-NewtonMethod.jl`: a Newton-method inversion example.
- `L6/04-inversion-GaussNewton.jl`: a Gauss-Newton inversion example.
- `L6/05-problem-setup-hard.jl`: shared setup for the harder, more non-linear comparison case.
- `L6/06-data-generation-hard.jl`: shared noisy synthetic data for the hard comparison case.
- `L6/07-inversion-GaussNewton-hard.jl`: local Gauss-Newton on the hard comparison case.
- `L6/08-inversion-PSO-hard.jl`: Particle Swarm Optimisation on the hard comparison case.
- `L6/09-inversion-VFSA-hard.jl`: Very Fast Simulated Annealing on the hard comparison case.

## Getting Started with Julia

These Julia scripts only use Julia standard libraries such as `LinearAlgebra`, `Printf`, and `Random`, so you do not need to install extra Julia packages to run the `.jl` files.

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
GEOM2057/L6
```


### 3. Run the Julia scripts

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

Each script prints the iteration history and final inversion result in the terminal.

## Suggested Workflow for Students

1. Run `01-problem-setup-derivatives-etc.jl` to understand the toy inverse problem setup.
2. Run `03-inversion-NewtonMethod.jl` and inspect the printed iteration history.
3. Run `04-inversion-GaussNewton.jl` and compare it against Newton's method.
4. Run `05-problem-setup-hard.jl` and `06-data-generation-hard.jl` to inspect the shared hard case and the noisy observations used by all three comparison methods.
5. Run `07-inversion-GaussNewton-hard.jl`, `08-inversion-PSO-hard.jl`, and `09-inversion-VFSA-hard.jl` to compare local and global search on the same hard inverse problem.
