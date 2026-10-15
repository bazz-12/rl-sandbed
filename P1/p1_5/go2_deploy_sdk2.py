"""
go2_deploy_sdk2.py - [P1.5] Run the state machine (go2_fsm.py) on the Go2, or in unitree_mujoco, through
unitree_sdk2_python's low-level interface.

!!! NOT TESTED ON HARDWARE. Written from Unitree's example/go2/low_level/go2_stand_example.py and checked only
!!! through the Genesis bench (go2_fsm_sim.py), which exercises go2_fsm.py but not this file.
!!! Go step by step: 1) unitree_mujoco, 2) robot hung on a stand, feet clear of the ground, 3) on the ground.

    python go2_deploy_sdk2.py --policy logs/go2-p15/exported --net enp3s0   # robot: interface wired to the Go2
    python go2_deploy_sdk2.py --policy logs/go2-p15/exported --sim          # unitree_mujoco (domain 1, "lo")

What it does
- On the robot, it first releases Unitree's sport mode (MotionSwitcherClient, as go2_stand_example.py does), so
  the onboard controller stops sending its own commands. The robot lies down first (StandDown).
- Every 2 ms it reads rt/lowstate (joints, IMU, gamepad), calls Go2StateMachine.update() and writes rt/lowcmd.
- It starts in damping. Gamepad: L1+R2 stand up, L1+A policy, L1+R1 sit down, L1+Y damping (software e-stop).
- Damping if no new robot state arrived for 50 ms, and for 1 s on Ctrl+C before exiting.
"""

import argparse
import signal
import struct
import threading
import time

import numpy as np

from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.go2.sport.sport_client import SportClient
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.utils.thread import RecurrentThread

from go2_fsm import FsmConfig, Go2StateMachine, Mode, Policy

# Motor index of each joint in rt/lowcmd and rt/lowstate (unitree_legged_const.py LegID: FR_0 = 0 ... RL_2 = 11)
SDK_JOINTS = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
              "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
POS_STOP_F, VEL_STOP_F = 2.146e9, 16000.0  # unitree_legged_const.py: "no position / velocity target"
KEY_BITS = ["R1", "L1", "start", "select", "R2", "L2", "F1", "F2", "A", "B", "X", "Y", "up", "right", "down", "left"]
CONTROL_DT = 0.002  # [s] as go2_stand_example.py
STATE_TIMEOUT_S = 0.05


def parse_remote(data):
    """wireless_remote (40 bytes) -> (pressed buttons, (lx, ly, rx, ry)). Layout as unitree_rl_gym
    deploy_real/common/remote_controller.py: key bits in the uint16 at bytes 2-3, then floats lx (4-7), rx (8-11),
    ry (12-15), ly (20-23)."""
    raw = bytes(data)
    keys = struct.unpack("<H", raw[2:4])[0]
    buttons = {name for i, name in enumerate(KEY_BITS) if keys >> i & 1}
    lx, rx, ry = struct.unpack("<3f", raw[4:16])
    (ly,) = struct.unpack("<f", raw[20:24])
    return buttons, (lx, ly, rx, ry)


def release_sport_mode():
    """go2_stand_example.py: lie down with the sport controller, then release it."""
    sc = SportClient()
    sc.SetTimeout(5.0)
    sc.Init()
    msc = MotionSwitcherClient()
    msc.SetTimeout(5.0)
    msc.Init()
    status, result = msc.CheckMode()
    while result["name"]:
        sc.StandDown()
        msc.ReleaseMode()
        status, result = msc.CheckMode()
        time.sleep(1)


class Go2Deploy:
    def __init__(self, fsm: Go2StateMachine, sdk_idx):
        self.fsm, self.sdk_idx = fsm, sdk_idx  # sdk_idx[i]: motor index of policy joint i
        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0], self.cmd.head[1] = 0xFE, 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for m in self.cmd.motor_cmd:  # 20 slots, as InitLowCmd(): servo mode, no target, no gains
            m.mode, m.q, m.dq, m.kp, m.kd, m.tau = 0x01, POS_STOP_F, VEL_STOP_F, 0.0, 0.0, 0.0
        self.lock = threading.Lock()
        self.state, self.state_time = None, 0.0
        self.t0 = time.monotonic()
        self.stop_requested = False
        self.error = False
        self.publisher = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.publisher.Init()
        self.subscriber = ChannelSubscriber("rt/lowstate", LowState_)
        self.subscriber.Init(self.on_state, 10)

    def on_state(self, msg: LowState_):
        with self.lock:
            self.state, self.state_time = msg, time.monotonic()

    def step(self):
        now = time.monotonic()
        with self.lock:
            s, age = self.state, now - self.state_time
        if s is None:  # no command before the first robot state
            return
        t = now - self.t0
        q = np.array([s.motor_state[j].q for j in self.sdk_idx], dtype=float)
        dq = np.array([s.motor_state[j].dq for j in self.sdk_idx], dtype=float)
        gyro = np.array(s.imu_state.gyroscope, dtype=float)  # base frame [rad/s]
        quat = np.array(s.imu_state.quaternion, dtype=float)  # (w, x, y, z)
        buttons, sticks = parse_remote(s.wireless_remote)
        try:
            if age > STATE_TIMEOUT_S:
                self.fsm.emergency_damping(t, f"no robot state for {1000 * age:.0f} ms")
            if self.stop_requested:
                self.fsm.emergency_damping(t, "Ctrl+C")
            q_des, dq_des, kp, kd, tau = self.fsm.update(t, q, dq, gyro, quat, buttons, sticks)
        except Exception as e:  # never leave the motors without a command: damping
            if not self.error:
                print(f"error in the state machine, damping: {e!r}")
            self.error = True
            zeros = np.zeros_like(q)
            q_des, dq_des, kp, kd, tau = q, zeros, zeros, np.full_like(q, self.fsm.cfg.passive_kd), zeros
        for i, j in enumerate(self.sdk_idx):
            m = self.cmd.motor_cmd[j]
            m.q, m.dq, m.kp, m.kd, m.tau = float(q_des[i]), float(dq_des[i]), float(kp[i]), float(kd[i]), float(tau[i])
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.publisher.Write(self.cmd)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True, help="folder written by go2_export.py (policy.pt, deploy_cfg.json)")
    p.add_argument("--net", type=str, default=None, help="network interface wired to the robot, e.g. enp3s0")
    p.add_argument("--domain", type=int, default=0, help="DDS domain id (robot 0, unitree_mujoco 1)")
    p.add_argument("--sim", action="store_true", help="unitree_mujoco: domain 1, interface lo, no sport mode")
    args = p.parse_args()
    if args.sim:
        args.domain, args.net = 1, args.net or "lo"

    policy = Policy(args.policy)
    sdk_idx = [SDK_JOINTS.index(name) for name in policy.cfg["joint_names"]]
    fsm = Go2StateMachine(policy, FsmConfig())

    print("WARNING: low-level control. Robot hung on a stand for the first run, clear area, finger on L1+Y.")
    input("Press Enter to continue...")
    if args.net:
        ChannelFactoryInitialize(args.domain, args.net)
    else:
        ChannelFactoryInitialize(args.domain)
    if not args.sim:
        release_sport_mode()

    node = Go2Deploy(fsm, sdk_idx)
    thread = RecurrentThread(interval=CONTROL_DT, target=node.step, name="go2_fsm")
    thread.Start()
    print("Running in damping mode. L1+R2 stand up | L1+A policy | L1+R1 sit down | L1+Y damping | Ctrl+C quit")
    try:
        while True:
            time.sleep(1.0)
            if node.state is None:
                print("waiting for rt/lowstate ...")
            elif fsm.mode is Mode.POLICY:
                print(f"policy | command vx {fsm.command[0]:+.2f} vy {fsm.command[1]:+.2f} wz {fsm.command[2]:+.2f}")
    except KeyboardInterrupt:
        signal.signal(signal.SIGINT, signal.SIG_IGN)  # a second Ctrl+C must not cut the damping phase short
        node.stop_requested = True
        time.sleep(1.0)  # keep sending damping for 1 s
        thread.Wait(0.1)
        print("stopped in damping mode (the robot is limp: restart sport mode from the app or power-cycle)")


if __name__ == "__main__":
    main()
