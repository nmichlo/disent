
"""
This file is an alias to the wandb cli that first sets the
temporary directory to a different folder `/tmp/<user>/tmp`,
in case `/tmp` has been polluted or you don't have the correct
access rights to modify files.

- I am not sure why we need to do this, it is probably a bug with
  wandb (or even tempfile) not respecting the `TMPDIR`, `TEMP` and
  `TMP` environment variables which when set should do the same as
  below? According to the tempdir docs:
  https://docs.python.org/3/library/tempfile.html#tempfile.gettempdir
"""

# wandb_cli.py
if __name__ == '__main__':
    import os
    import tempfile

    # generate the temporary directory from the user
    temp_dir = f'/tmp/{os.environ["USER"]}/wandb'
    print(f'[PATCHING:] tempfile.tempdir={repr(temp_dir)}')

    # we need to patch tempdir before we can import wandb
    assert tempfile.tempdir is None
    os.makedirs(temp_dir, exist_ok=True)
    tempfile.tempdir = temp_dir

    # taken from wandb.__main__
    from wandb.cli.cli import cli
    cli(prog_name="python -m wandb")
