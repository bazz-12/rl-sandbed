"""
test_go2_deploy.py - [S2R] Checks of the deploy-side logic that need neither a robot nor a simulator.

    python test_go2_deploy.py                         # state machine, inputs, kinematics, log format
    python test_go2_deploy.py --policy logs/<exp>/exported   # the same on your exported policy (else a synthetic cfg)
    python test_go2_deploy.py --dds                   # + the deploy loop's error path over DDS (loopback, domain 99)

Every check prints one line; the script exits with an error at the first failure.
"""

import argparse
import os
import sys
import tempfile
import time

import numpy as np

import go2_kinematics as K
import go2_log
from go2_fsm import FsmConfig, Go2StateMachine, Mode, Policy
from go2_inputs import (KEY_BITS, PROFILES, SCENARIOS, CommandProfile, JointTest, Scenario, encode_remote,
                        parse_remote)

NAMES = ["FR_hip_joint", "FR_thigh_joint", "FR_calf_joint", "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
         "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint", "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint"]
LEVEL = np.array([1.0, 0.0, 0.0, 0.0])


def synthetic_cfg():
    """What go2_export.make_deploy_cfg() writes for the go2_train.py config."""
    default = [0.0, 0.8, -1.5, 0.0, 0.8, -1.5, 0.0, 1.0, -1.5, 0.0, 1.0, -1.5]
    lim = {"hip": (-1.0472, 1.0472), "F": (-1.5708, 3.4907), "R": (-0.5236, 4.5379), "calf": (-2.7227, -0.83776)}
    limits = [lim["hip"] if "hip" in n else lim["calf"] if "calf" in n else lim[n[0]] for n in NAMES]
    return {"joint_names": NAMES, "default_joint_angles": default, "kp": 20.0, "kd": 0.5, "action_scale": 0.25,
            "clip_actions": 100.0, "obs_scales": {"lin_vel": 2.0, "ang_vel": 0.25, "dof_pos": 1.0, "dof_vel": 0.05},
            "commands_scale": [2.0, 2.0, 0.25], "num_commands": 3,
            "command_ranges": {"lin_vel_x": [-1.0, 1.0], "lin_vel_y": [-0.5, 0.5], "ang_vel": [-1.0, 1.0]},
            "lin_deadzone": 0.2, "ang_deadzone": 0.2, "base_height_target": 0.3, "recurrent": False,
            "policy_dt": 0.02, "torque_limits": [23.7, 23.7, 45.43] * 4, "dof_pos_limits": limits}


def ok(msg):
    print(f"ok   {msg}")


class Bench:
    """Drives a state machine with a synthetic robot that follows its targets exactly."""

    def __init__(self, policy_dir, **cfg):
        self.policy = Policy(policy_dir) if policy_dir else Policy(deploy_cfg=synthetic_cfg())
        self.fsm = Go2StateMachine(self.policy, FsmConfig(**cfg), verbose=False)
        self.t, self.dt = 0.0, 0.002
        self.q = self.fsm.lie_pose.copy()
        self.quat = LEVEL.copy()

    def run(self, seconds, buttons=(), sticks=(0.0, 0.0, 0.0, 0.0), press_s=0.1, follow=True):
        out = None
        for k in range(int(round(seconds / self.dt))):
            b = buttons if k * self.dt < press_s else ()
            out = self.fsm.update(self.t, self.q, np.zeros(12), np.zeros(3), self.quat, b, sticks)
            if follow and self.fsm.mode is not Mode.PASSIVE:
                self.q = out[0].copy()
            self.t += self.dt
        return out

    def stand(self):
        self.run(3.0, ("L1", "R2"))
        assert self.fsm.mode is Mode.STAND_UP and self.fsm.standing_ready(self.t), self.fsm.events

    def policy_mode(self):
        self.stand()
        self.run(1.0, ("L1", "A"))
        assert self.fsm.mode is Mode.POLICY and self.fsm.policy_running, self.fsm.events


def test_inputs():
    for b in [set(), {"L1", "R2"}, set(KEY_BITS)]:
        bb, st = parse_remote(encode_remote(b, (0.1, -0.5, 0.25, 1.0)))
        assert bb == b and np.allclose(st, (0.1, -0.5, 0.25, 1.0))
    ok("remote bytes: encode -> parse round trip")
    for name in PROFILES:
        pr = CommandProfile(name, 4 if name == "height" else 3)
        assert pr.command(pr.duration) is None and pr.segment(pr.duration - 1e-6) == len(pr.segments) - 1
    ok(f"{len(PROFILES)} profiles: segments and end")
    jt = JointTest(NAMES)
    off = np.array([jt.offset(t) for t in np.arange(0.0, jt.duration, 0.002)])
    assert np.abs(off).max() <= 0.15 + 1e-9 and np.abs(off[-1]).max() < 1e-9
    assert np.abs(np.diff(off[(np.arange(len(off)) * 0.002 > jt.chirp_t0)], axis=0)).max() / 0.002 < 4.0
    ok(f"joint test: {jt.duration:.1f} s, offsets within 0.15 rad, chirp under 4 rad/s, ends at zero")
    for name in SCENARIOS:
        Scenario(name, CommandProfile("fwd_back"), 21.7)
    ok(f"{len(SCENARIOS)} scenarios compile")


def test_fsm(policy_dir):
    b = Bench(policy_dir)
    b.run(0.5)
    assert b.fsm.mode is Mode.PASSIVE
    b.run(1.0, ("L1", "A"))
    assert b.fsm.mode is Mode.PASSIVE, "L1+A must be ignored in damping"
    b.stand()
    ok("damping -> stand-up; L1+A ignored before standing")
    b.run(1.0, ("L1", "A"))
    assert b.fsm.mode is Mode.POLICY
    q_des, dq_des, kp, kd, tau = b.run(0.2)
    assert np.allclose(kp, 20.0) and np.allclose(kd, 0.5), "policy gains"
    ok("policy mode after the 0.5 s hand-over, with the policy gains")
    b.run(0.5, ("L1", "Y"))
    assert b.fsm.mode is Mode.PASSIVE
    q_des, dq_des, kp, kd, tau = b.run(0.01)
    assert np.all(kp == 0) and np.allclose(kd, 3.0)
    ok("L1+Y -> damping (kp 0, kd 3)")

    # [S2R] a crooked stand (reversed motor, obstacle) must not hand over to the policy
    b = Bench(policy_dir)
    b.stand()
    a = np.radians(25.0)
    b.quat = np.array([np.cos(a / 2), np.sin(a / 2), 0.0, 0.0])  # 25 deg roll
    b.run(1.0, ("L1", "A"))
    assert b.fsm.mode is Mode.STAND_UP and any("L1+A refused" in e[1] for e in b.fsm.events), b.fsm.events
    ok("L1+A refused when the base tilts more than 20 deg after the stand-up")

    # held buttons do not re-trigger: L1+R2 held through the stand-up must not restart it
    b = Bench(policy_dir)
    b.run(3.0, ("L1", "R2"), press_s=3.0)
    assert sum("stand_up" in e[1] for e in b.fsm.events) == 1
    ok("button combos trigger on the press, not while held")

    # tilt trip
    b = Bench(policy_dir)
    b.policy_mode()
    b.quat = np.array([np.cos(0.6), np.sin(0.6), 0.0, 0.0])  # 1.2 rad roll
    b.run(0.01)
    assert b.fsm.mode is Mode.PASSIVE and "tilt" in b.fsm.events[-1][1]
    ok("tilt over 1 rad in policy mode -> damping")

    # joint outside its range
    b = Bench(policy_dir)
    b.policy_mode()
    b.q[2] = -0.3  # FR calf, range [-2.72, -0.84] + 0.2 margin
    b.run(0.01, follow=False)
    assert b.fsm.mode is Mode.PASSIVE and "outside" in b.fsm.events[-1][1]
    ok("joint 0.2 rad past its range -> damping")

    # torque trip: a joint held far from its target for longer than 100 ms
    b = Bench(policy_dir)
    b.stand()
    b.q = b.fsm.stand_pose.copy()
    b.q[1] += 2.0  # FR thigh 2 rad off: 80 x 2 = 160 Nm asked, 6.8 x the limit
    b.run(0.05, follow=False)
    assert b.fsm.mode is Mode.STAND_UP, "no trip before torque_trip_s"
    b.run(0.1, follow=False)
    assert b.fsm.mode is Mode.PASSIVE and "torque" in b.fsm.events[-1][1], b.fsm.events[-1]
    ok("1.5 x the torque limit for 100 ms -> damping")

    # non-finite policy output
    b = Bench(policy_dir)
    b.policy_mode()
    b.policy.step = lambda *a, **k: (_ for _ in ()).throw(FloatingPointError("policy output is not finite"))
    b.run(0.05)
    assert b.fsm.mode is Mode.PASSIVE and "not finite" in b.fsm.events[-1][1]
    ok("non-finite policy output -> damping")

    # target clipping
    b = Bench(policy_dir)
    b.stand()
    b.fsm.lie_pose[2] = -3.5  # an impossible calf target
    b.run(0.5, ("L1", "R1"))
    q_des = b.run(2.5)[0]
    assert q_des[2] >= b.fsm.q_lo[2] - 1e-9 and b.fsm.clamp_count > 0
    ok("targets clipped to the URDF range")


def test_profile(policy_dir):
    b = Bench(policy_dir)
    b.fsm.load_profile(CommandProfile("slow", b.fsm.num_commands))
    b.policy_mode()
    b.run(0.2, ("L1", "X"))
    b.run(2.5)
    assert b.fsm.profile_t0 is not None and np.allclose(b.fsm.command[:3], (0.25, 0, 0)), b.fsm.command
    ok("L1+X starts the profile; segment 1 commands vx 0.25")
    b.run(0.1, sticks=(0.0, 0.8, 0.0, 0.0))
    assert b.fsm.profile_t0 is None and "operator" in b.fsm.events[-1][1]
    ok("a stick past half way stops the profile")
    b.run(0.2, ("L1", "X"))
    b.run(19.0)
    assert b.fsm.profile_t0 is None and np.allclose(b.fsm.command, 0.0)
    ok("the profile ends with a zero command")
    b = Bench(policy_dir)
    b.policy_mode()
    b.run(0.2, ("L1", "X"))
    assert b.fsm.profile_t0 is None and "no command profile" in b.fsm.events[-1][1]
    ok("L1+X without --profile is ignored")


def test_joint_test(policy_dir):
    b = Bench(policy_dir)
    b.stand()
    b.run(0.2, ("L1", "B"))
    assert b.fsm.mode is Mode.STAND_UP and "not allowed" in b.fsm.events[-1][1]
    ok("L1+B ignored unless the joint test is allowed")
    b = Bench(policy_dir, allow_joint_test=True)
    b.stand()
    b.run(0.2, ("L1", "B"))
    assert b.fsm.mode is Mode.JOINT_TEST
    b.run(b.fsm.joint_test_duration())
    assert b.fsm.mode is Mode.STAND_UP and "joint test done" in b.fsm.events[-1][1]
    ok(f"joint test runs {b.fsm.joint_test_duration():.1f} s and returns to the stand hold")


def test_policy_observation(policy_dir):
    if not policy_dir:
        return
    p = Policy(policy_dir)
    rng = np.random.default_rng(0)
    q = np.asarray(p.cfg["default_joint_angles"]) + 0.1 * rng.standard_normal(12)
    obs = p.observation(rng.standard_normal(3), LEVEL, q, rng.standard_normal(12), np.zeros(p.cfg["num_commands"]))
    assert obs.shape == (p.n_obs,) and np.allclose(obs[3:6], (0, 0, -1))
    p.reset()
    a1 = p.step(np.zeros(3), LEVEL, q, np.zeros(12), np.zeros(p.cfg["num_commands"]))
    p.reset()
    a2 = p.step(np.zeros(3), LEVEL, q, np.zeros(12), np.zeros(p.cfg["num_commands"]))
    assert np.allclose(a1, a2), "reset() must clear the memory"
    ok(f"exported policy: {p.n_obs} observations, deterministic after reset(), inference {1e3 * p.last_infer_s:.2f} ms")


def test_kinematics():
    rng = np.random.default_rng(1)
    q = np.tile([0.0, 0.8, -1.5], 4) + 0.2 * rng.standard_normal((5, 12))
    J = K.jacobians(q)
    eps = 1e-6
    for j in range(3):
        d = np.zeros((5, 12))
        d[:, j::3] = eps
        Jn = (K.foot_positions(q + d) - K.foot_positions(q - d)) / (2 * eps)
        assert np.abs(J[..., j] - Jn).max() < 1e-7
    ok("leg Jacobians match finite differences")
    F = np.array([[0.0, 0.0, 37.0]] * 4)[None]
    tau = -np.einsum("nkji,nkj->nki", J[:1], F)  # tau = -J^T F
    assert np.allclose(K.foot_forces(q[:1], tau.reshape(1, 12)), F, atol=1e-6)
    ok("foot forces from torques invert tau = -J^T F")
    stand = np.tile([0.0, 0.8, -1.5], 4)[None]
    h = K.base_height(stand, LEVEL[None], np.ones((1, 4), bool))[0]
    assert 0.25 < h < 0.40, h
    ok(f"default pose: base {h:.3f} m above flat ground")
    fz = np.zeros((200, 4))
    fz[20:80], fz[120:180] = 60.0, 60.0  # two 120 ms stances per foot ...
    fz[50:52] = 0.0  # ... one bounce of 2 samples inside the first
    fz[100:103] = 60.0  # ... one 3-sample spike in the swing between
    st = K.debounce(K.contacts(fz), 8)
    td = (st[1:] & ~st[:-1]).sum(0)
    assert np.all(td == 2) and st[50:52].all() and not st[100:103].any(), td
    ok("contact debounce: a bounce in stance and a spike in swing are not steps")


def test_sim_models():
    """[S2R] go2_mujoco_headless.py's motor, sensor and fault models (pure numpy parts)."""
    import go2_mujoco_headless as H

    assert H.motor_index("FR_calf") == 2 and H.motor_index("RL_hip_joint") == 9 and H.motor_index("11") == 11
    assert H.parse_windows("12:0.2,20:1") == [(12.0, 12.2), (20.0, 21.0)]
    rng = np.random.default_rng(0)
    qj, dqj, tj = rng.normal(size=12), rng.normal(size=12), rng.normal(size=12)
    healthy = H.MotorMap()
    assert healthy.identity and all(np.allclose(a, b) for a, b in zip(healthy.to_motor(qj, dqj, tj), (qj, dqj, tj)))
    m = H.MotorMap(swap=(1, 2), flip=[4], offset={7: 0.05})
    qm, dqm, _ = m.to_motor(qj, dqj, tj)
    assert np.isclose(qm[1], qj[2]) and np.isclose(qm[2], qj[1]) and np.isclose(qm[4], -qj[4])
    assert np.isclose(qm[7], qj[7] + 0.05) and np.isclose(dqm[4], -dqj[4])
    tau_j = m.to_joint(np.arange(12.0))
    assert tau_j[2] == 1.0 and tau_j[1] == 2.0 and tau_j[4] == -4.0
    ok("simulator faults: a swap, a reversed motor and an encoder offset map state and torque consistently")
    cmd = np.zeros((5, 12))
    cmd[0], cmd[2] = 0.5, 20.0  # q* 0.5 rad, kp 20
    act = H.Actuators(H.MotorMap(), per_step=False, delay_s=0.01)
    act.receive(0.0, cmd)
    assert act.torques(0.005, np.zeros(12), np.zeros(12)) is None  # not yet due
    assert np.allclose(act.torques(0.010, np.zeros(12), np.zeros(12)), 10.0)
    assert np.allclose(act.torques(0.012, np.full(12, 0.5), np.zeros(12)), 10.0), "hold: torque kept"
    act = H.Actuators(H.MotorMap(), per_step=True, strength=0.5)
    act.receive(0.0, cmd)
    assert np.allclose(act.torques(0.0, np.zeros(12), np.zeros(12)), 5.0)
    assert np.allclose(act.torques(0.002, np.full(12, 0.5), np.zeros(12)), 0.0), "per step: re-evaluated"
    big = cmd.copy()
    big[0] = 10.0
    act.receive(0.0, big)
    assert np.allclose(act.torques(0.004, np.zeros(12), np.zeros(12)), 0.5 * H.TORQUE_LIMITS), "clipped at S x limit"
    ok("simulator motors: command delay, held torque vs PD every step, strength and torque limits")
    dl = H.DelayLine(0.1)
    for k in range(50):
        dl.put(k * 0.002, k)
    assert dl.get(0.098 - 0.010) == 44 and dl.get(-1.0) == 0
    ok("simulator state delay: the reading from D ms ago")


def test_log():
    with tempfile.TemporaryDirectory() as d:
        rec = go2_log.Recorder(os.path.join(d, "x.npz"), "robot", max_s=1.0, ctrl_hz=100.0, n_obs=45, n_cmd=3)
        for i in range(250):  # more rows than kept: the ring buffer keeps the last ones
            rec.append("ctrl", t=i * 0.01, q=np.full(12, i), mode=2)
        rec.append("pol", t=0.0, obs=np.ones(45), action=np.ones(12), cmd=np.zeros(3), infer=1e-4, seg=-1)
        rec.event(0.5, "hello")
        lg = go2_log.load(rec.save(extra_meta={"deploy_cfg": synthetic_cfg()}))
        assert lg.ctrl["t"][-1] == 2.49 and np.all(np.diff(lg.ctrl["t"]) > 0) and lg.events == [[0.5, "hello"]]
    ok("log: ring buffer keeps the newest rows in order; meta and events round trip")


def test_dds_robot_services():
    """[S2R] go2_deploy_sdk2.release_sport_mode() against go2_mujoco_headless.py's fake robot services."""
    import go2_deploy_sdk2 as dep
    import go2_mujoco_headless as H

    switcher = H.start_robot_services(lambda msg: None)
    t0 = time.monotonic()
    dep.release_sport_mode(timeout_s=15.0)
    assert switcher.released and switcher.mode == ""
    ok(f"sport-mode release against the fake services ({time.monotonic() - t0:.1f} s): the robot path runs")


def test_dds_error_path(policy_dir):
    """go2_deploy_sdk2.Go2Deploy: an exception in the control step must still write a damping command."""
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize

    import go2_deploy_sdk2 as dep

    ChannelFactoryInitialize(99, "lo")
    policy = Policy(policy_dir) if policy_dir else Policy(deploy_cfg=synthetic_cfg())
    fsm = Go2StateMachine(policy, FsmConfig(), verbose=False)
    node = dep.Go2Deploy(fsm, list(range(12)))
    written = []
    node.publisher.Write = lambda cmd: written.append([(m.q, m.kp, m.kd) for m in cmd.motor_cmd[:12]])
    node._step = lambda: 1 / 0
    print("     (the traceback below is expected)")
    node.step()
    assert written and all(kp == 0.0 and kd == fsm.cfg.passive_kd and q == dep.POS_STOP_F for q, kp, kd in written[-1])
    ok("an exception in the control step still writes a damping command")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", type=str, default=None, help="exported policy folder (default: synthetic config)")
    p.add_argument("--dds", action="store_true", help="also test the deploy loop's error path (needs the SDK)")
    args = p.parse_args()
    t0 = time.time()
    test_inputs()
    test_fsm(args.policy)
    test_profile(args.policy)
    test_joint_test(args.policy)
    test_policy_observation(args.policy)
    test_kinematics()
    test_sim_models()
    test_log()
    if args.dds:
        test_dds_error_path(args.policy)
        test_dds_robot_services()
    print(f"all checks passed ({time.time() - t0:.1f} s)")


if __name__ == "__main__":
    sys.exit(main())
