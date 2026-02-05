#!/usr/bin/env python3
import subprocess
from pathlib import Path
from itertools import product

# -----------------------
# Parameter lists
# -----------------------

mpi_processes_list = [16]
nodes_list = [4]
subdomain_refinement_list = [2]
ml = 7

# -----------------------
# Static configuration
# -----------------------

job_base_name = "mc"
binary = "./mantlecirculation"
job_dir = Path("job_scripts")

static_slurm_header = """#!/bin/bash -l
#
#SBATCH --gres=gpu:h100:4
#SBATCH --partition=h100
#SBATCH --time=00:20:00
#SBATCH --export=NONE
#SBATCH --mail-user=fabian.boehm@fau.de
#SBATCH --mail-type=BEGIN

"""

static_runtime_args = [
    "--outdir-overwrite",
    "--max-timesteps 10",
    "--t-end 1.0",
    "--refinement-level-mesh-min 2"
]

# -----------------------
# Create job directory
# -----------------------

job_dir.mkdir(exist_ok=True)

generated_scripts = []

# -----------------------
# Generate scripts
# -----------------------

for mpi_procs, nodes, sub_ref in zip(
    mpi_processes_list,
    nodes_list,
    subdomain_refinement_list,
):
    job_name = f"{job_base_name}_np{mpi_procs}_n{nodes}_sdr{sub_ref}_ml{ml}"
    script_path = job_dir / f"{job_name}.sh"

    with script_path.open("w") as f:
        f.write(static_slurm_header)
        f.write(f"#SBATCH --nodes={nodes}\n")
        f.write(f"#SBATCH --job-name={job_name}\n\n")
        f.write(f"\nunset SLURM_EXPORT_ENV\nmodule load openmpi/5.0.5-nvhpc24.11-cuda cmake \n ")
        cmd = [
            "mpirun",
            f"-np {mpi_procs}",
            binary,
            *static_runtime_args,
            f"--refinement-level-subdomains {sub_ref}",
            f"--refinement-level-mesh-max {ml}"
        ]

        f.write(" ".join(cmd) + "\n")

    script_path.chmod(0o755)
    generated_scripts.append(script_path)

# -----------------------
# Submit jobs
# -----------------------

submitted = 0

for script in generated_scripts:
    try:
        result = subprocess.run(
            ["sbatch", script],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"Submitted {script.name}: {result.stdout.strip()}")
        submitted += 1
        script.unlink()  # remove script after submission
    except subprocess.CalledProcessError as e:
        print(f"ERROR submitting {script.name}")
        print(e.stderr)

# -----------------------
# Cleanup directory
# -----------------------

if not any(job_dir.iterdir()):
    job_dir.rmdir()

print(f"\nDone. Submitted {submitted} jobs.")

