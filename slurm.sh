#!/bin/bash
#SBATCH --job-name = qmio_tes
#SBATCH --partition = qpu

source .venv/bin/activate
python qmio_script.py

