"""Main CLI module for NumbaCC."""

import io
from contextlib import redirect_stderr, redirect_stdout

import click

from nbcc.compiler import compile as _compile
from nbcc.compiler import compile_shared_lib, compile_to_mlir


class SpecialGroup(click.Group):
    """Custom Click Group that allows fallback to a default command."""

    def get_command(self, ctx, cmd_name):
        rv = super().get_command(ctx, cmd_name)
        if rv is not None:
            return rv

        # Check if this might be a file path - if so, treat as default compile command
        import os

        if os.path.exists(cmd_name) or "." in cmd_name:
            # This looks like a file, so invoke default compile behavior
            ctx.params["input_file"] = cmd_name
            return self.get_command(ctx, "compile")
        return None


@click.group(cls=SpecialGroup, invoke_without_command=True)
@click.pass_context
def main(ctx):
    """NumbaCC - Numba-like compiler for SPy.

    \b
    Usage:
      nbcc <input_file> <output_file>    # Compile to binary (default)
      nbcc compile <input_file> <output_file>  # Explicit compile to binary
      nbcc shared <input_file> <output_file>   # Compile to shared library
      nbcc mlir <input_file>                   # Generate and print MLIR

    \b
    Examples:
      nbcc input.spy output              # Compile to binary
      nbcc shared input.spy output.so    # Compile to shared library
      nbcc compile input.spy output      # Explicit binary compilation
      nbcc mlir input.spy                # Print MLIR to terminal
    """
    if ctx.invoked_subcommand is None:
        # Show help when no arguments provided
        click.echo(ctx.get_help())


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
def compile(input_file, output_file):
    """Compile SPy source to binary executable (default command).

    INPUT_FILE: Path to the SPy source file to compile
    OUTPUT_FILE: Path for the compiled binary executable
    """
    _compile(str(input_file), str(output_file))


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
def shared(input_file, output_file):
    """Compile SPy source to shared library.

    INPUT_FILE: Path to the SPy source file to compile
    OUTPUT_FILE: Path for the compiled shared library (.so/.dylib/.dll)
    """
    compile_shared_lib(str(input_file), str(output_file))


@main.command()
@click.argument("input_file", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_file", type=click.Path(dir_okay=False))
@click.option(
    "--quiet",
    "-q",
    is_flag=True,
    help="Suppress debug output, show only final MLIR",
)
@click.option(
    "--backend",
    "-b",
    type=click.Choice(["cpu", "cutile"]),
    default="cpu",
    help="Backend to use for compilation",
)
def mlir(input_file, output_file, quiet, backend):
    """Generate and print MLIR for SPy source.

    INPUT_FILE: Path to the SPy source file to compile to MLIR

    Use --backend to select compilation backend (llvm, cutile, gpu).
    Use --quiet to suppress debug output and show only final MLIR.
    """


    # Backend selection logic - TODO: Implement backend-specific compilation
    match backend:
        case "cpu":
            from nbcc.mlir_backend.backend import Backend
            be_type = Backend
        case "cutile":
            from nbcc.cutile_backend.backend import CuTileBackend
            be_type = CuTileBackend
        case _:
            raise NotImplementedError(f"{backend!r} is not available")

    if quiet:
        # Capture and suppress debug output, only show final MLIR
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            module = compile_to_mlir(str(input_file), be_type=be_type)
    else:
        # Show all debug output as normal
        module = compile_to_mlir(str(input_file), be_type=be_type)

    # Print the final MLIR
    click.echo("\n" + "=" * 60)
    click.echo("FINAL MLIR:")
    click.echo("=" * 60)
    output = module.operation.get_asm(enable_debug_info=False)
    click.echo(output)
    with open(output_file, "w") as fout:
        print(output, file=fout)


if __name__ == "__main__":
    main()
