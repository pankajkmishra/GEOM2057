# GEOM2057 Lecture 6

This repository keeps only the Julia teaching scripts and the README tracked for Lecture 6.

The repository currently includes:

- `L6/01-problem-description.jl`: shared definitions and a problem overview.
- `L6/02-data-generation.jl`: synthetic data generation for the toy inverse problem.
- `L6/03-inversion-NewtonMethod.jl`: a Newton-method inversion example.
- `L6/04-inversion-GaussNewton.jl`: a Gauss-Newton inversion example.

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

Run the problem description:

```powershell
julia .\01-problem-description.jl
```

Run the Newton inversion example:

```powershell
julia .\03-inversion-NewtonMethod.jl
```

Run the Gauss-Newton inversion example:

```powershell
julia .\04-inversion-GaussNewton.jl
```

Each script prints the iteration history and final inversion result in the terminal.

## Suggested Workflow for Students

1. Run `01-problem-description.jl` to understand the toy inverse problem setup.
2. Run `03-inversion-NewtonMethod.jl` and inspect the printed iteration history.
3. Run `04-inversion-GaussNewton.jl` and compare it against Newton's method.
