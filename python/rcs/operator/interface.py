from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
import threading
import copy
from time import sleep
from typing import Protocol
import numpy as np
import gymnasium as gym

from rcs.envs.base import ArmWithGripper, ControlMode, RelativeTo
from rcs.utils import SimpleFrameRate

@dataclass
class TeleopCommands:
    """Semantic commands decoupled from specific hardware buttons."""
    record: bool = False
    success: bool = False
    failure: bool = False
    reset_origin: bool = False

@dataclass(kw_only=True)
class BaseOperatorConfig:
    env_frequency: int = 30

class BaseOperator(ABC, threading.Thread):
    """Interface for an operator device"""
    
    # Define this as a class attribute so it can be accessed without instantiating
    control_mode: tuple[ControlMode, RelativeTo]

    def __init__(self, env: gym.Env, config: BaseOperatorConfig):
        super().__init__()
        self.config = config
        self.env = env
        self.reset_lock = threading.Lock()
        self._exit_requested = False
        
        # State for semantic commands
        self._commands = TeleopCommands()
        self._cmd_lock = threading.Lock()

    def consume_commands(self) -> TeleopCommands:
        """Returns the current commands and resets them to False (edge-triggered)."""
        with self._cmd_lock:
            cmds = copy.copy(self._commands)
            self._commands.record = False
            self._commands.success = False
            self._commands.failure = False
            self._commands.reset_origin = False
            return cmds

    def reset_operator_state(self):
        """Hook for subclasses to reset their internal poses/offsets on env reset."""
        pass

    def run(self):
        """Read out hardware, set states and process buttons."""
        raise NotImplementedError()

    # TODO: support multiple robots
    def get_action(self) -> dict[str, ArmWithGripper]:
        """Returns the action dictionary to step the environment."""
        raise NotImplementedError()

    def stop(self):
        self._exit_requested = True
        self.join()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    def environment_step_loop(self):
        rate_limiter = SimpleFrameRate(self.config.env_frequency, "env loop")
        while True:
            if self._exit_requested:
                break
                
            # 1. Process Meta-Commands
            cmds = self.consume_commands()
            
            if cmds.record:
                print("Command: Start Recording")
                self.env.get_wrapper_attr("start_record")()
                
            if cmds.success:
                print("Command: Success! Resetting env...")
                with self.reset_lock:
                    self.env.get_wrapper_attr("success")()
                    sleep(1) # sleep to let the robot reach the goal
                    self.env.reset()
                    self.reset_operator_state()
                    
            elif cmds.failure:
                print("Command: Failure! Resetting env...")
                with self.reset_lock:
                    self.env.reset()
                    self.reset_operator_state()
            
            # if cmds.reset_origin:
            #     print("Command: Resetting origin...")
            #     # env lock
            #     for robot in self.config.robot_keys:
            #         self.env.envs[robot].set_origin_to_current()


            # 2. Step the Environment
            with self.reset_lock:
                actions = self.get_action()
                if actions: # Only step if actions are provided
                    self.env.step(actions)
                    
            rate_limiter()


class CompositeOperator(BaseOperator):
    def __init__(self, env, motion_operator: BaseOperator, command_operator: BaseOperator):
        # We don't need a specific config for the composite itself, 
        # so we just pass a default one to the base class
        super().__init__(env, BaseOperatorConfig())
        
        self.motion_op = motion_operator
        self.command_op = command_operator
        
        # Inherit the control mode from the motion operator (e.g., GELLO)
        self.control_mode = self.motion_op.control_mode

    def start(self):
        """Start the background threads for both hardware readers."""
        self.motion_op.start()
        self.command_op.start()

    def stop(self):
        """Stop both hardware readers."""
        self.motion_op.stop()
        self.command_op.stop()
        self._exit_requested = True

    def get_action(self):
        """Fetch the physical movements from the motion operator (GELLO)."""
        return self.motion_op.get_action()

    def consume_commands(self) -> TeleopCommands:
        """Fetch the meta-commands (record/success/fail) from the command operator (Pedal)."""
        # If both devices can send commands, you could logically OR them together here.
        # But in this case, only the pedal sends commands.
        return self.command_op.consume_commands()

    def reset_operator_state(self):
        """Pass the reset hook down to the operators."""
        self.motion_op.reset_operator_state()
        self.command_op.reset_operator_state()
        
    def run(self):
        # The base class requires this, but the sub-operators handle their own run loops.
        pass