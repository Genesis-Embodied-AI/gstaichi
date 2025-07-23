# type: ignore

import argparse
import math
import os
import runpy
import shutil
import subprocess
import sys
import timeit
from collections import defaultdict
from functools import wraps
from pathlib import Path

import rich
from colorama import Fore
from rich.console import Console

from gs_taichi._lib import core as _ti_core
from gs_taichi._lib import utils
from gs_taichi.lang import impl
from gs_taichi.tools import diagnose


def timer(func):
    """Function decorator to benchmark a function running time."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start = timeit.default_timer()
        result = func(*args, **kwargs)
        elapsed = timeit.default_timer() - start
        print(f">>> Running time: {elapsed:.2f}s")
        return result

    return wrapper


def registerableCLI(cls):
    """Class decorator to register methods with @register into a set."""
    cls.registered_commands = set([])
    for name in dir(cls):
        method = getattr(cls, name)
        if hasattr(method, "registered"):
            cls.registered_commands.add(name)
    return cls


def register(func):
    """Method decorator to register CLI commands."""
    func.registered = True
    return func


@registerableCLI
class TaichiMain:
    def __init__(self, test_mode: bool = False):
        self.banner = f"\n{'*' * 43}\n**      Taichi Programming Language      **\n{'*' * 43}"
        print(self.banner)

        print(self._get_friend_links())

        parser = argparse.ArgumentParser(description="Taichi CLI", usage=self._usage())
        parser.add_argument("command", help="command from the above list to run")

        # Flag for unit testing
        self.test_mode = test_mode

        self.main_parser = parser

    @timer
    def __call__(self):
        # Print help if no command provided
        if len(sys.argv[1:2]) == 0:
            self.main_parser.print_help()
            return 1

        # Parse the command
        args = self.main_parser.parse_args(sys.argv[1:2])

        if args.command not in self.registered_commands:  # pylint: disable=E1101
            # TODO: do we really need this?
            if args.command.endswith(".py"):
                TaichiMain._exec_python_file(args.command)
            else:
                print(f"{args.command} is not a valid command!")
                self.main_parser.print_help()
            return 1

        return getattr(self, args.command)(sys.argv[2:])

    @staticmethod
    def _get_friend_links():
        return (
            "\n"
            "Docs:   https://docs.taichi-lang.org/\n"
            "GitHub: https://github.com/taichi-dev/taichi/\n"
            "Forum:  https://forum.taichi.graphics/\n"
        )

    def _usage(self) -> str:
        """Compose deterministic usage message based on registered_commands."""
        # TODO: add some color to commands
        msg = "\n"
        space = 20
        for command in sorted(self.registered_commands):  # pylint: disable=E1101
            msg += f"    {command}{' ' * (space - len(command))}|-> {getattr(self, command).__doc__}\n"
        return msg

    @staticmethod
    def _exec_python_file(filename: str):
        """Execute a Python file based on filename."""
        # TODO: do we really need this?
        subprocess.call([sys.executable, filename] + sys.argv[1:])

    @staticmethod
    def _get_examples_dir() -> Path:
        """Get the path to the examples directory."""

        root_dir = utils.package_root
        examples_dir = Path(root_dir) / "examples"
        return examples_dir

    @staticmethod
    def _get_available_examples() -> set:
        """Get a set of all available example names."""
        examples_dir = TaichiMain._get_examples_dir()
        all_examples = examples_dir.rglob("*.py")
        all_example_names = {f.stem: f.parent for f in all_examples}
        return all_example_names

    @staticmethod
    def _example_choices_type(choices):
        def support_choice_with_dot_py(choice):
            if choice.endswith(".py") and choice.split(".")[0] in choices:
                # try to find and remove python file extension
                return choice.split(".")[0]
            return choice

        return support_choice_with_dot_py

    @register
    def example(self, arguments: list = sys.argv[2:]):
        """Run an example by name (or name.py)"""

        def colormap(index, name):
            from colorsys import hls_to_rgb  # pylint: disable=C0415

            x = (ord(name[0].upper()) - 64.0) / 26.0
            r, g, b = hls_to_rgb(x, 0.4, 1.0)
            r = hex(int(r * 255) % 16)[2:]
            g = hex(int(g * 255) % 16)[2:]
            b = hex(int(b * 255) % 16)[2:]
            return f"{index}: [#{r}{r}{g}{g}{b}{b}]{name}"

        console = Console()
        table = rich.table.Table(
            box=rich.box.HORIZONTALS,
            show_header=False,
            header_style="bold #2070b2",
            title="[bold][#3fdda4]TAICHI[#f8e020] EXAMPLES",
        )

        ncols = 3
        choices = TaichiMain._get_available_examples()
        nrows, rem = divmod(len(choices), ncols)
        if rem > 0:
            nrows += 1
        names = sorted(choices.keys())
        for k in range(nrows):
            table.add_row(*[colormap(j, names[j]) for j in range(k, len(choices), nrows)])

        parser = argparse.ArgumentParser(prog="ti example", description=f"{self.example.__doc__}")
        parser.add_argument(
            "name",
            type=TaichiMain._example_choices_type(choices.keys()),
            choices=sorted(choices.keys()),
            help=console.print(table),
            nargs="?",
            default=None,
            metavar="name",
        )
        parser.add_argument(
            "-p",
            "--print",
            required=False,
            dest="print",
            action="store_true",
            help="Print example source code instead of running it",
        )
        parser.add_argument(
            "-P",
            "--pretty-print",
            required=False,
            dest="pretty_print",
            action="store_true",
            help="Like --print, but print in a rich format with line numbers",
        )
        parser.add_argument(
            "-s",
            "--save",
            required=False,
            dest="save",
            action="store_true",
            help="Save source code to current directory instead of running it",
        )

        # TODO: Pass the arguments to downstream correctly(#3216).
        args = parser.parse_args(arguments)

        examples_dir = TaichiMain._get_examples_dir()
        example_name = args.name
        if example_name is None:
            try:
                index = input(f"Please input the example index (between 0 and {len(names)}): ")
                while not index.isdigit() or int(index) >= len(names):
                    index = input(f"Example [{index}] does not exist. Please try again: ")
                example_name = names[int(index)]
            except KeyboardInterrupt as e:
                print("\nCancelled by user, exiting...")
                return 1

        target = str((examples_dir / choices[example_name] / f"{example_name}.py").resolve())
        # path for examples needs to be modified for implicit relative imports
        sys.path.append(str((examples_dir / choices[example_name]).resolve()))

        # Short circuit for testing
        if self.test_mode:
            return args

        if args.save:
            print(f"Saving example {example_name} to current directory...")
            shutil.copy(target, ".")
            return 0

        if args.pretty_print:
            syntax = rich.syntax.Syntax.from_path(target, line_numbers=True)
            console = Console()
            console.print(syntax)
            return 0

        if args.print:
            with open(target) as f:
                print(f.read())
            return 0

        print(f"Running example {example_name} ...")

        runpy.run_path(target, run_name="__main__")

    @staticmethod
    @register
    def changelog(arguments: list = sys.argv[2:]):
        """Display changelog of current version"""
        changelog_md = os.path.join(utils.package_root, "CHANGELOG.md")
        with open(changelog_md) as f:
            print(f.read())

    @staticmethod
    @register
    def release(arguments: list = sys.argv[2:]):
        """Make source code release"""
        raise RuntimeError("TBD")

    @staticmethod
    @register
    def doc(arguments: list = sys.argv[2:]):
        """Build documentation"""
        raise RuntimeError("TBD")

    @staticmethod
    @register
    def format(arguments: list = sys.argv[2:]):
        """Reformat modified source files"""
        raise RuntimeError("Please run `pre-commit run -a` instead")

    @staticmethod
    @register
    def format_all(arguments: list = sys.argv[2:]):
        """Reformat all source files"""
        raise RuntimeError("Please run `pre-commit run -a` instead")

    @staticmethod
    def _display_benchmark_regression(xd, yd, args):
        def parse_dat(file):
            _dict = {}
            with open(file) as f:
                for line in f.readlines():
                    try:
                        a, b = line.strip().split(":")
                    except:
                        continue
                    b = float(b)
                    if abs(b % 1.0) < 1e-5:  # codegen_*
                        b = int(b)
                    _dict[a.strip()] = b
            return _dict

        def parse_name(file):
            if file[0:5] == "test_":
                return file[5:-4].replace("__test_", "::", 1)
            if file[0:10] == "benchmark_":
                return "::".join(reversed(file[10:-4].split("__arch_")))
            raise Exception(f"bad benchmark file name {file}")

        def get_dats(directory):
            _list = []
            for x in os.listdir(directory):
                if x.endswith(".dat"):
                    _list.append(x)
            _dict = {}
            for x in _list:
                name = parse_name(x)
                path = os.path.join(directory, x)
                _dict[name] = parse_dat(path)
            return _dict

        spec = args.files
        single_line = spec and len(spec) == 1
        xs, ys = get_dats(xd), get_dats(yd)
        scatter = defaultdict(list)
        for name in reversed(sorted(set(xs.keys()).union(ys.keys()))):
            file, func = name.split("::")
            u, v = xs.get(name, {}), ys.get(name, {})
            ret = ""
            for key in set(u.keys()).union(v.keys()):
                if spec and key not in spec:
                    continue
                a, b = u.get(key, 0), v.get(key, 0)
                if a == 0:
                    if b == 0:
                        res = 1.0
                    else:
                        res = math.inf
                else:
                    res = b / a
                scatter[key].append(res)
                if res == 1:
                    continue
                if not single_line:
                    ret += f"{key:<30}"
                res -= 1
                color = Fore.RESET
                if res > 0:
                    color = Fore.RED
                elif res < 0:
                    color = Fore.GREEN
                if isinstance(a, float):
                    a = f"{a:>7.2}"
                else:
                    a = f"{a:>7}"
                if isinstance(b, float):
                    b = f"{b:>7.2}"
                else:
                    b = f"{b:>7}"
                ret += f"{Fore.MAGENTA}{a}{Fore.RESET} -> "
                ret += f"{Fore.CYAN}{b} {color}{res:>+9.1%}{Fore.RESET}\n"
            if ret != "":
                print(f'{file + "::" + func:_<58}', end="")
                if not single_line:
                    print("")
                print(ret, end="")
                if not single_line:
                    print("")

    @staticmethod
    def _get_benchmark_baseline_dir():
        return os.path.join(_ti_core.get_repo_dir(), "benchmarks", "baseline")

    @staticmethod
    def _get_benchmark_output_dir():
        return os.path.join(_ti_core.get_repo_dir(), "benchmarks", "output")

    @register
    def regression(self, arguments: list = sys.argv[2:]):
        """Display benchmark regression test result"""
        parser = argparse.ArgumentParser(prog="ti regression", description=f"{self.regression.__doc__}")
        parser.add_argument("files", nargs="*", help="Test file(s) to be run for benchmarking")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode:
            return args

        baseline_dir = TaichiMain._get_benchmark_baseline_dir()
        output_dir = TaichiMain._get_benchmark_output_dir()
        TaichiMain._display_benchmark_regression(baseline_dir, output_dir, args)

        return None

    @register
    def baseline(self, arguments: list = sys.argv[2:]):
        """Archive current benchmark result as baseline"""
        parser = argparse.ArgumentParser(prog="ti baseline", description=f"{self.baseline.__doc__}")
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode:
            return args

        baseline_dir = TaichiMain._get_benchmark_baseline_dir()
        output_dir = TaichiMain._get_benchmark_output_dir()
        shutil.rmtree(baseline_dir, True)
        shutil.copytree(output_dir, baseline_dir)
        print(f"[benchmark] baseline data saved to {baseline_dir}")

        return None

    @staticmethod
    @register
    def test(self, arguments: list = sys.argv[2:]):
        raise RuntimeError("ti test is deprecated. Please run `python tests/run_tests.py` instead.")

    @register
    def run(self, arguments: list = sys.argv[2:]):
        """Run a single script"""
        parser = argparse.ArgumentParser(prog="ti run", description=f"{self.run.__doc__}")
        parser.add_argument(
            "filename",
            help="A single (Python) script to run with Taichi, e.g. render.py",
        )
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode:
            return args

        runpy.run_path(args.filename)

        return None

    @register
    def debug(self, arguments: list = sys.argv[2:]):
        """Debug a single script"""
        parser = argparse.ArgumentParser(prog="ti debug", description=f"{self.debug.__doc__}")
        parser.add_argument(
            "filename",
            help="A single (Python) script to run with debugger, e.g. render.py",
        )
        args = parser.parse_args(arguments)

        # Short circuit for testing
        if self.test_mode:
            return args

        _ti_core.set_core_trigger_gdb_when_crash(True)
        os.environ["TI_DEBUG"] = "1"

        runpy.run_path(args.filename, run_name="__main__")

        return None

    @staticmethod
    @register
    def diagnose(arguments: list = sys.argv[2:]):
        """System diagnose information"""
        diagnose.main()

    @staticmethod
    @register
    def repl(arguments: list = sys.argv[2:]):
        """Start Taichi REPL / Python shell with 'import gs_taichi as ti'"""

        def local_scope():
            try:
                import IPython  # pylint: disable=C0415

                IPython.embed()
            except ImportError:
                import code  # pylint: disable=C0415

                __name__ = "__console__"  # pylint: disable=W0622
                code.interact(local=locals())

        local_scope()

    @staticmethod
    @register
    def lint(arguments: list = sys.argv[2:]):
        """Run pylint checker for the Python codebase of Taichi"""
        # TODO: support arguments for lint specific files
        # parser = argparse.ArgumentParser(prog='ti lint', description=f"{self.lint.__doc__}")
        # args = parser.parse_args(arguments)

        options = [os.path.dirname(__file__)]

        from multiprocessing import cpu_count  # pylint: disable=C0415

        threads = min(8, cpu_count())
        options += ["-j", str(threads)]

        # http://pylint.pycqa.org/en/latest/user_guide/run.html
        # TODO: support redirect output to lint.log
        import pylint  # pylint: disable=C0415

        pylint.lint.Run(options)

    @register
    def module(self, arguments: list = sys.argv[2:]):
        """Taichi module tools"""
        from gs_taichi import _ti_module  # pylint: disable=C0415

        _ti_module._main(arguments)

    @staticmethod
    @register
    def cache(arguments: list = sys.argv[2:]):
        """Manage the offline cache files manually"""

        def clean(cmd_args, parser):
            parser.add_argument(
                "-p",
                "--offline-cache-file-path",
                dest="offline_cache_file_path",
                default=impl.default_cfg().offline_cache_file_path,
            )
            args = parser.parse_args(cmd_args)
            path = os.path.abspath(args.offline_cache_file_path)
            count = _ti_core.clean_offline_cache_files(path)
            print(f"Deleted {count} offline cache files in {path}")

        # TODO(PGZXB): Provide more tools to manage the offline cache files
        subcmds_map = {"clean": (clean, "Clean all offline cache files in given path")}

        def print_help():
            print("usage: ti cache <command> [<args>]")
            for name, value in subcmds_map.items():
                _, description = value
                print(f"\t{name}\t|-> {description}")

        if not arguments:
            print_help()
            return

        subcmd = arguments[0]
        if subcmd not in subcmds_map:
            print(f"'ti cache {subcmd}' is not a valid command!")
            print_help()
            return

        func, description = subcmds_map[subcmd]
        parser = argparse.ArgumentParser(prog=f"ti cache {subcmd}", description=description)
        func(cmd_args=arguments[1:], parser=parser)


def main():
    cli = TaichiMain()
    return cli()


if __name__ == "__main__":
    sys.exit(main())
