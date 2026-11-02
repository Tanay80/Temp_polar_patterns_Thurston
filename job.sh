#!/bin/bash
#SBATCH --job-name=tpp
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=0
#SBATCH --time=48:00:00
#SBATCH --output=runall.out
#SBATCH --error=runall.err
#SBATCH --exclusive
#SBATCH --partition=compute
#SBATCH --qos=compute

cd ~/tanay/
source venv/bin/activate
cd tpp/
python runall.py
