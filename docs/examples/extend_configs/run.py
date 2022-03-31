import os.path

from experiment.run import hydra_experiment

# run the experiment and append the new config search path
# - for example:
#   $ python3 run.py dataset=E--pseudorandom
if __name__ == '__main__':
    hydra_experiment(search_dirs_prepend=os.path.abspath(os.path.join(__file__, '../config')))
