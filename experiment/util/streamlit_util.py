#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~
#  MIT License
#
#  Copyright (c) 2021 Nathan Juraj Michlo
#
#  Permission is hereby granted, free of charge, to any person obtaining a copy
#  of this software and associated documentation files (the "Software"), to deal
#  in the Software without restriction, including without limitation the rights
#  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#  copies of the Software, and to permit persons to whom the Software is
#  furnished to do so, subject to the following conditions:
#
#  The above copyright notice and this permission notice shall be included in
#  all copies or substantial portions of the Software.
#
#  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#  SOFTWARE.
#  ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~

import sys


# ========================================================================= #
# Streamlit Runner                                                          #
# ========================================================================= #


IS_WRAPPED = False


def run_streamlit(python_file):
    """
    IMPORTANT: import this function from a seperate file!

    Use this function by placing a call to it at the top of your
    check for __main__

    >>> # from run_streamlit import run_streamlit
    >>>
    >>> if __name__ == '__main__':
    >>>     run_streamlit(__file__)
    >>>     st.title('Hello World!')
    """

    # do not wrap if streamlit is already running!
    global IS_WRAPPED
    if IS_WRAPPED:
        return True

    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    import click
    import streamlit.cli

    @click.group()
    def run_streamlit():
        pass

    # For some reason I cant get streamlit to work without this subcommand?
    @run_streamlit.command('streamlit')
    @streamlit.cli.configurator_options
    def run_streamlit_subcommand(**kwargs):
        global IS_WRAPPED
        IS_WRAPPED = True
        streamlit.cli._apply_config_options_from_cli(kwargs)
        streamlit.cli._main_run(python_file, args)

    # ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~ #

    # swap out arguments and keep arguments, first is the filename
    args = sys.argv[1:]
    sys.argv = [sys.argv[0], 'streamlit']

    # run streamlit
    run_streamlit()
    exit(0)


# ========================================================================= #
# END                                                                       #
# ========================================================================= #
