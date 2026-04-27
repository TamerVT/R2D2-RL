from pathlib import Path
from typing import Annotated

import typer
from rcs.envs.storage_wrapper import StorageWrapper
from rcs.sim.replayer import replay as replay_dataset

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


@app.command("replay")
def replay(
    dataset: Annotated[
        Path,
        typer.Argument(
            exists=True,
            help="Parquet dataset directory to replay.",
        ),
    ],
    headless: Annotated[bool, typer.Option(help="Whether to run without GUI.")] = True,
    frequency: Annotated[int, typer.Option(help="Simulation frequency to use during replay.")] = 30,
    relative_to: Annotated[
        str,
        typer.Option(help="RelativeTo enum name: CONFIGURED_ORIGIN, LAST_STEP, or NONE."),
    ] = "CONFIGURED_ORIGIN",
    scene: Annotated[
        str,
        typer.Option(help="Python expression that evaluates to a scene instance."),
    ] = "env_configs.EmptyWorldFR3Duo()",
    task_cfg: Annotated[
        str,
        typer.Option(help="Python expression that evaluates to a task config."),
    ] = 'env_tasks.PickTaskConfig(robot_name="right")',
):
    replay_dataset(
        dataset=dataset,
        headless=headless,
        frequency=frequency,
        relative_to=relative_to,
        scene=scene,
        task_cfg=task_cfg,
    )


if __name__ == "__main__":
    app()
