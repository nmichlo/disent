import sys
from contextlib import contextmanager
import pytest
import os
from glob import glob
from disent.util import _set_test_run


@contextmanager
def no_stdout():
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')
    yield
    sys.stdout = old_stdout


ROOT_DIR = os.path.abspath(__file__ + '/../..')


@pytest.mark.parametrize("module", [
    os.path.relpath(path, ROOT_DIR).replace('/', '.')[:-3]
    for path in glob(os.path.join(ROOT_DIR, 'docs/examples/**.py'))
])
def test_docs_examples(capsys, module):
    # make sure everything doesnt take 5 years to run
    _set_test_run()
    # run all the files in the examples folder
    import importlib
    with no_stdout():
        importlib.import_module(module)
