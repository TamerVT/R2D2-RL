from pathlib import Path
from typing import Annotated

import typer
from rcs.envs.storage_wrapper import StorageWrapper

app = typer.Typer()


@app.command()
def consolidate(
    path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=False,
            dir_okay=True,
            help="The root directory of the parquet dataset to consolidate.",
        ),
    ]
):
    """
    Consolidates a fragmented Parquet dataset into larger files.

    This is useful if the recording process crashed or was interrupted,
    leaving many small files behind.
    """
    typer.echo(f"Starting consolidation for: {path}")

    StorageWrapper.consolidate(str(path), schema=None)

    typer.echo("Done.")


if __name__ == "__main__":
    app()
