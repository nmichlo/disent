import os.path

from experiment.run import hydra_experiment

# This python script is functionally equivalent to run.sh

# run the experiment and append the new config search path
# - for example:
#   $ python3 run.py dataset=E--pseudorandom framework=E--si-betavae
if __name__ == "__main__":
    hydra_experiment(search_dirs_prepend=os.path.abspath(os.path.join(__file__, "../config")))
