import sys


# ========================================================================= #
# Streamlit Runner                                                          #
# ========================================================================= #


STREAMLIT_COMMAND = 'streamlit'
WRAP_COMMAND = 'wrapped'


def _launch_streamlit(python_file, args: list = None):
    import click
    import streamlit.cli

    @click.group()
    def main_streamlit():
        pass

    # For some reason I cant get streamlit to work without this subcommand?
    @main_streamlit.command(STREAMLIT_COMMAND)
    @streamlit.cli.configurator_options
    def run_streamlit_subcommand(**kwargs):
        streamlit.cli._apply_config_options_from_cli(kwargs)
        streamlit.cli._main_run(python_file, args if args else [])

    main_streamlit()


def run_streamlit(python_file):
    """
    This function requires that no command line arguments are being passed to the program.
    # TODO: change to check last command line argument

    Use this function by placing a call to it at the top of your file
    that you want to run with streamlit

    >>> if __name__ == '__main__':
    >>>     from experiment.util.streamlit_util import run_streamlit
    >>>     run_streamlit(__file__)
    """

    # append streamlit command by default
    if len(sys.argv) == 1:
        sys.argv.append(STREAMLIT_COMMAND)
    command = sys.argv[1]

    # launch streamlit or hydra depending on command
    if command == STREAMLIT_COMMAND:
        _launch_streamlit(python_file, [WRAP_COMMAND])
        exit(0)
    elif command == WRAP_COMMAND:
        return True
    else:
        raise KeyError(f'Unknown: {command=} | {sys.argv}')


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
