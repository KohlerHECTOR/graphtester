#!/bin/bash
#SBATCH --job-name=random_ds   # nom du job
#SBATCH --ntasks=1             # Nombre total de processus MPI
#SBATCH --ntasks-per-node=1
#SBATCH --account=aes@cpu
# Dans le vocabulaire Slurm "multithread" fait référence à l'hyperthreading.
#SBATCH --hint=nomultithread   # 1 processus MPI par coeur physique (pas d'hyperthreading)
#SBATCH --time=20:00:00        # Temps d’exécution maximum demande (HH:MM:SS)
#SBATCH --output=%x_%A_%a.out  # Nom du fichier de sortie contenant l'ID et l'indice
#SBATCH --error=%x_%A_%a.out   # Nom du fichier d'erreur (ici commun avec la sortie)
#SBATCH --array=0-12

# on se place dans le répertoire de soumission
cd /lustre/fswork/projects/rech/aes/uma45ed/WL-Express/graphtester/

# nettoyage des modules charges en interactif et herites par defaut
module purge

# chargement des modules
module load python/3.12.7
module load cuda/12.8.0

# echo des commandes lancées
set -x

# Execution du binaire "mon_exe" avec des donnees differentes pour chaque travail
# La valeur de ${SLURM_ARRAY_TASK_ID} est differente pour chaque travail.
source wl_env/bin/activate
chmod +x jobs-graphtester/job_$((SLURM_ARRAY_TASK_ID)).sh
srun bash jobs-graphtester/job_$((SLURM_ARRAY_TASK_ID)).sh