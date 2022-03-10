import argparse
import itertools
import sys

import numpy as np

import disent.util.strings.colors as c
from hydra.core.override_parser.overrides_parser import OverridesParser
from hydra.core.override_parser.types import Override
from hydra.core.override_parser.types import Sweep


def _iter_value_strings(override: Override) -> str:
    if isinstance(override.value(), Sweep):
        yield from override.sweep_string_iterator()
    else:
        yield override.get_value_element_as_str()


if __name__ == '__main__':
    # sys.argv.extend(['--line-numbers', '--base', 'PYTHONPATH=. python3 experiment/run.py', '--overrides', 'framework=betavae,adavae_os', 'dataset=X--xysquares,cars3d,shapes3d,dsprites,smallnorb'])

    # From: `hydra._internal.utils.get_args_parser`
    parser = argparse.ArgumentParser(add_help=True, description="Hydra Permutations")
    parser.add_argument("--overrides", nargs="*", default=(),  help="Any key=value arguments parsed using Hydra Config. Values are treated as a sweep and all permutations are generated on the output instead.")
    parser.add_argument("--base", type=str, default='',        help='The base command that will be added to the output.')
    parser.add_argument("--debug", action='store_true',        help='If debug information should be displayed.')
    parser.add_argument("--line-number-start",  type=int, default=1,    help='The starting number to use when labeling lines')
    parser.add_argument("--line-numbers",       action='store_true',    help='If line numbers should be prepended as environment variables.')
    parser.add_argument("--line-number-format", type=str, default=None, help='The format for prefixing line numbers, default is `CMD_NUM={}`')
    parser.add_argument("--no-color", action='store_true',     help='Disable ANSI colors in the output.')
    parser.add_argument("--no-align", action='store_true',     help='Disable alignment of different arguments in the output')
    args = parser.parse_args()

    # From: ` hydra._internal.config_loader_impl.ConfigLoaderImpl._load_configuration_impl`
    parser = OverridesParser.create()
    parsed_overrides = parser.parse_overrides(overrides=args.overrides)

    # color object
    if args.no_color:
        c, _c = argparse.Namespace(), c
        for k in dir(_c):
            setattr(c, k, '')
    else:
        import disent.util.strings.colors as c

    # DEBUG: input information
    if args.debug:
        print(f'{c.lGRY}COMMAND:{c.RST}   {c.RED}{repr(args.base)}{c.RST}')
        print(f'{c.lGRY}OVERRIDES:{c.RST} {c.YLW}{repr(args.overrides)}{c.RST}')
        print()

    # get permutation strings
    override_orig_strings, override_permutations = [], []
    for override in parsed_overrides:
        override_orig_strings.append(f'{c.lMGT}{override.input_line}{c.RST}')
        prefix = override.get_key_element()
        override_permutations.append([f'{c.lGRY}{prefix}{c.lGRY}={c.lYLW}{postfix}{c.RST}' for postfix in _iter_value_strings(override)])

    # DEBUG: parsed override information
    if args.debug:
        for orig_str, elem_strs in zip(override_orig_strings, override_permutations):
            print(f'{c.lMGT}*{c.RST} {orig_str}')
            for elem_str in elem_strs:
                print(f'{c.lGRY}| {elem_str}')
        if parsed_overrides:
            print()

    # get stats
    num_override_elems = [len(elems) for elems in override_permutations]
    num_permutations = int(np.prod(num_override_elems))

    # DEBUG: separator
    if args.debug:
        print(f'{c.lGRY}{"="*100}{c.RST}')
        print(f'{c.lGRY}GENERATED {num_permutations} COMMANDS ({"x".join(str(n) for n in num_override_elems)}){c.RST}')
        print(f'{c.lGRY}{"="*100}{c.RST}')
        print()

    # if we need to print the line numbers
    if args.line_numbers and (not args.no_align):
        num_digits = int(np.log10(max(1, num_permutations))) + 1
    else:
        num_digits = 0
    # if we need to normalise the length of everything so that the output is aligned
    if args.no_align:
        elem_lengths = [0 for _ in override_permutations]
    else:
        elem_lengths = [max(len(elem) for elem in elems) for elems in override_permutations]

    # OUTPUT: print permutations
    for i, elements in enumerate(itertools.product(*override_permutations)):
        command_parts = []
        # prepend the line numbers
        if args.line_numbers:
            num = i + args.line_number_start
            if args.line_number_format is None:
                line_num = f'CMD_NUM={num:<{num_digits}d}'
            else:
                line_num = args.line_number_format.format(num)
            command_parts.append(f'{c.lGRY}{line_num}{c.RST}')
        # prepend the base command
        if args.base:
            command_parts.append(f'{c.lRED}{args.base}{c.RST}')
        # append the arguments
        if elements:
            command_parts.append(" ".join(f'{elem:<{l}s}' for elem, l in zip(elements, elem_lengths)))
        # done!
        print(' '.join(command_parts))
