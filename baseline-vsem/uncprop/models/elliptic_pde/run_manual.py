#!/projectnb/dietzelab/arober/Bayesian-inference-with-surrogates/.venv/bin/python -u
#$ -N run_manual
#$ -P gpsurr
#$ -j y
#$ -l h_rt=12:00:00
#$ -l mem_per_core=12G
#$ -pe omp 1

import sys
from datetime import datetime

from uncprop.models.elliptic_pde.runner import main_manual

timestamp = datetime.now()

print(f'Executable: {sys.executable}', flush=True)
print('Timestamp:', timestamp.strftime("%Y-%m-%d %H:%M:%S"))

n_design = 4
rep_idx = [2, 2]
write_to_log_file = True

main_manual(experiment_name='pde_test_run',
            n_design=n_design, 
            rep_idx=rep_idx, 
            write_to_log_file=write_to_log_file)