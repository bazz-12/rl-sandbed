"""
go2_commands.py - velocity-command conventions shared by training and deployment.

[P1.1] The same zeroing rule is applied when commands are sampled in training (go2_env.py) and when gamepad
sticks are turned into commands on the robot, so the deployed policy never receives a command it has not seen.
No Genesis import: copy this file to the deployment PC as is. Works on torch tensors and numpy arrays.
"""

LIN_DEADZONE = 0.2   # [m/s]   |(vx, vy)| below this -> (0, 0)  (legged_gym _resample_commands uses 0.2)
ANG_DEADZONE = 0.2   # [rad/s] |wz| below this -> 0


def apply_command_deadzone(cmd, lin_deadzone=LIN_DEADZONE, ang_deadzone=ANG_DEADZONE):
    """Zero small commands. cmd: [..., 3] = (vx, vy, wz), or [..., 4] with the body-height offset dh [P1.6],
    which passes through unchanged. Returns a new array/tensor of the same type."""
    out = cmd * 1.0                                             # copy, works for torch and numpy
    small_lin = (out[..., 0] ** 2 + out[..., 1] ** 2) < lin_deadzone ** 2
    small_ang = abs(out[..., 2]) < ang_deadzone
    out[..., 0] = out[..., 0] * ~small_lin
    out[..., 1] = out[..., 1] * ~small_lin
    out[..., 2] = out[..., 2] * ~small_ang
    return out


def sticks_to_command(forward, left, turn_left, max_cmd=(1.0, 0.5, 1.0), stick_deadzone=0.1,
                      lin_deadzone=LIN_DEADZONE, ang_deadzone=ANG_DEADZONE):
    """Gamepad sticks in [-1, 1] -> (vx, vy, wz) command.

    Map your gamepad axes so that pushing forward / left / turn-left gives +1 (check the sign convention of
    your gamepad driver). max_cmd should match the training command ranges.
    1) stick deadzone, rescaled so the command is continuous past it (removes stick drift),
    2) scaling to the command ranges,
    3) the training deadzone, so tiny commands become an exact zero (= stand still).
    """
    def stick(s):
        s = max(-1.0, min(1.0, float(s)))
        mag = max(abs(s) - stick_deadzone, 0.0) / (1.0 - stick_deadzone)
        return mag if s >= 0 else -mag

    cmd = [stick(forward) * max_cmd[0], stick(left) * max_cmd[1], stick(turn_left) * max_cmd[2]]
    import numpy as np
    return apply_command_deadzone(np.asarray(cmd, dtype=np.float32), lin_deadzone, ang_deadzone)
