from abc import ABC
import copy
from dataclasses import dataclass, field
import threading
from time import sleep
import gymnasium as gym

from rcs.envs.base import ArmWithGripper, ControlMode, RelativeTo
from rcs.sim.sim import Sim
from rcs.utils import SimpleFrameRate


@dataclass
class TeleopCommands:
    """Semantic commands decoupled from specific hardware buttons."""

    record: bool = False
    success: bool = False
    failure: bool = False
    reset_origin_to_current: dict[str, bool] = field(default_factory=dict)


@dataclass(kw_only=True)
class BaseOperatorConfig:
    read_frequency: int = 30
    simulation: bool = True


class BaseOperator(ABC, threading.Thread):
    control_mode: tuple[ControlMode, RelativeTo]
    controller_names: list[str] = field(default=["left", "right"])

    def __init__(self, config: BaseOperatorConfig, sim: Sim | None = None):
        threading.Thread.__init__(self)
        self.config = config
        self.sim = sim

    def consume_commands(self) -> TeleopCommands:
        """Returns the current commands and resets them (edge-triggered). Must be thread-safe."""
        raise NotImplementedError()

    def reset_operator_state(self):
        """Hook for subclasses to reset their internal poses/offsets on env reset. Must be thread-safe."""

    def run(self):
        """Read out hardware, set states and process buttons."""
        raise NotImplementedError()

    def consume_action(self) -> dict[str, ArmWithGripper]:
        """Returns the action dictionary to step the environment. Must be thread-safe."""
        raise NotImplementedError()


class TeleopLoop:
    """Interface for an operator device"""

    # Define this as a class attribute so it can be accessed without instantiating
    control_mode: tuple[ControlMode, RelativeTo]

    def __init__(
        self,
        env: gym.Env,
        operator: BaseOperator,
        env_frequency: int = 30,
        key_translation: dict[str, str] | None = None,
    ):
        super().__init__()
        self.env = env
        self.operator = operator
        self._exit_requested = False
        self.env_frequency = env_frequency
        if key_translation is None:
            # controller to robot translation
            self.key_translation = {key: key for key in self.operator.controller_names}
        else:
            self.key_translation = key_translation

    def stop(self):
        self._exit_requested = True
        self.operator.join()

    def __enter__(self):
        self.operator.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def _translate_keys(self, actions):
        return {self.key_translation[key]: actions[key] for key in actions}

    def environment_step_loop(self):
        rate_limiter = SimpleFrameRate(self.env_frequency, "env loop")
        while True:
            if self._exit_requested:
                break

            # 1. Process Meta-Commands
            cmds = self.operator.consume_commands()

            if cmds.record:
                print("Command: Start Recording")
                self.env.get_wrapper_attr("start_record")()

            if cmds.success:
                print("Command: Success! Resetting env...")
                self.env.get_wrapper_attr("success")()
                sleep(1)  # sleep to let the robot reach the goal
                self.env.reset()
                self.operator.reset_operator_state()
                # consume new commands because of potential origin reset
                continue

            elif cmds.failure:
                print("Command: Failure! Resetting env...")
                self.env.reset()
                self.operator.reset_operator_state()
                # consume new commands because of potential origin reset
                continue

            for controller in cmds.reset_origin_to_current:
                if cmds.reset_origin_to_current[controller]:
                    robot = self.key_translation[controller]
                    print(f"Command: Resetting origin for {robot}...")
                    assert (
                        self.operator.control_mode[1] == RelativeTo.CONFIGURED_ORIGIN
                        # TODO the following is a dict and can thus not easily be used like this
                        # and self.env.get_wrapper_attr("relative_to") == RelativeTo.CONFIGURED_ORIGIN
                    ), "both robot env and operator must be configured to relative_to.CONFIGURED_ORIGIN"
                    self.env.get_wrapper_attr("envs")[robot].set_origin_to_current()

            # 2. Step the Environment
            actions = self.operator.consume_action()
            actions = self._translate_keys(actions)
            self.env.step(actions)

            rate_limiter()
