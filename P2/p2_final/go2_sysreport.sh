#!/usr/bin/env bash
# go2_sysreport.sh - one report of everything needed to check the Go2 sim-to-real setup after the move to Kubuntu.
# Read-only except for the smoke tests (they write logs/ and out/ in your project folder, as usual). No sudo.
#
#   bash go2_sysreport.sh <project folder> <exp>          e.g.  bash go2_sysreport.sh ~/projects/sandbed go2-p16
#
# Optional, if your names differ:  SANDBED_ENV=sandbed DEPLOY_ENV=go2deploy UM=~/unitree_mujoco SDK=~/unitree_sdk2_python
# QUICK=1 skips the Genesis smoke tests (a few minutes on first run: kernels compile).
# Writes ~/go2_sysreport_<date>.txt: send me that file. It holds your user name, host name, paths, local IP addresses and
# Wi-Fi connection name (MAC addresses are masked). Leaves logs/sysreport_smoke/ behind only if the smoke train fails.

PROJ="$(cd "${1:-$HOME/projects/sandbed}" 2>/dev/null && pwd || echo "${1:-$HOME/projects/sandbed}")"
EXP="${2:-}"
SANDBED_ENV="${SANDBED_ENV:-sandbed}"
DEPLOY_ENV="${DEPLOY_ENV:-go2deploy}"
UM="${UM:-$HOME/unitree_mujoco}"
SDK="${SDK:-$HOME/unitree_sdk2_python}"
DDS_SRC="${DDS_SRC:-$HOME/cyclonedds}"
OUT="$HOME/go2_sysreport_$(date +%Y%m%d_%H%M).txt"

sec() { printf '\n==================== %s ====================\n' "$*"; }
run() {  # run "<command>" [timeout s] : prints the command, its output (first 80 lines) and exit code
  printf '\n$ %s\n' "$1"
  timeout "${2:-60}" bash -o pipefail -c "$1" 2>&1 | head -n 80
  printf '[exit %s]\n' "${PIPESTATUS[0]}"
}
have() { command -v "$1" >/dev/null 2>&1; }

find_conda() {
  local base=""
  if have conda; then base="$(conda info --base 2>/dev/null)"; fi
  for d in "$base" "$HOME/miniforge3" "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/mambaforge" /opt/conda; do
    if [ -n "$d" ] && [ -f "$d/etc/profile.d/conda.sh" ]; then echo "$d"; return; fi
  done
}

in_env() {  # in_env <env> "<shell commands>" [timeout s] : runs in that conda env, from the project folder
  printf '\n$ [%s] %s\n' "$1" "$2"
  ( cd "$PROJ" 2>/dev/null; source "$CONDA_BASE/etc/profile.d/conda.sh" && conda activate "$1" && \
    timeout "${3:-120}" bash -o pipefail -c "$2" ) 2>&1 | tail -n 60
  printf '[exit %s]\n' "${PIPESTATUS[0]}"
}

main() {
  echo "go2_sysreport  $(date -Is)  project=$PROJ  exp=${EXP:-<none>}  envs=$SANDBED_ENV,$DEPLOY_ENV  UM=$UM  SDK=$SDK"

  sec "1. System"
  run "lsb_release -ds; uname -r; echo session=\$XDG_SESSION_TYPE desktop=\$XDG_CURRENT_DESKTOP display=\$DISPLAY wayland=\$WAYLAND_DISPLAY"
  run "plasmashell --version"
  run "lscpu | grep -E 'Model name|^CPU\\(s\\)|Thread|MHz'; free -h"
  run "df -hT '$PROJ' '$HOME' /"
  run "cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_driver /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor; powerprofilesctl get"
  run "cat /sys/class/power_supply/A*/online 2>/dev/null | sed 's/^/AC online: /'"

  sec "2. GPU and drivers"
  run "lspci -nnk | grep -EA3 'VGA|3D'"
  run "nvidia-smi" 30
  run "cat /proc/driver/nvidia/version; lsmod | grep -E '^nvidia|^nouveau'"
  run "prime-select query; mokutil --sb-state"
  run "ubuntu-drivers devices" 90
  run "glxinfo -B" 20
  run "nvidia-smi -q -d PERFORMANCE,POWER | grep -vE '^\s*$' | head -n 40" 30
  run "ls /usr/share/glvnd/egl_vendor.d/; ls -l /dev/dri"

  sec "3. Conda"
  CONDA_BASE="$(find_conda)"
  echo "conda base: ${CONDA_BASE:-NOT FOUND}"
  if [ -n "$CONDA_BASE" ]; then
    run "source '$CONDA_BASE/etc/profile.d/conda.sh'; conda --version; conda env list; conda config --show channels"
  fi
  run "grep -nE 'conda|CYCLONEDDS|MUJOCO_GL|PYOPENGL|CUDA|LD_LIBRARY_PATH|PYTHONPATH|alias python|starship' ~/.bashrc ~/.profile ~/.bash_profile ~/.bash_aliases 2>/dev/null"
  if [ -n "$CONDA_BASE" ]; then
    run "source '$CONDA_BASE/etc/profile.d/conda.sh'; for e in $SANDBED_ENV $DEPLOY_ENV; do echo \"--- \$e:\"; conda env config vars list -n \$e; done"
  fi
  run "echo '--- user site-packages (can shadow env packages unless PYTHONNOUSERSITE=1):'; ls -d ~/.local/lib/python3*/site-packages 2>/dev/null && ls ~/.local/lib/python3*/site-packages | head -20 || echo none"
  run "echo '--- disk used by conda:'; du -sh '${CONDA_BASE:-/nonexistent}'/pkgs '${CONDA_BASE:-/nonexistent}'/envs/* 2>/dev/null"

  if [ -n "$CONDA_BASE" ]; then
    sec "4. Training env: $SANDBED_ENV"
    in_env "$SANDBED_ENV" "which python; python - <<'EOF'
import sys, importlib.metadata as md
print('python', sys.version.split()[0], sys.executable)
for p in ('genesis-world', 'rsl-rl-lib', 'torch', 'numpy', 'tensorboard', 'gstaichi', 'quadrants', 'mujoco', 'matplotlib'):
    try: print(f'{p:14s}', md.version(p))
    except Exception: print(f'{p:14s}', '-')
import torch
print('torch cuda build', torch.version.cuda, '| cuda available', torch.cuda.is_available())
if torch.cuda.is_available():
    print('device', torch.cuda.get_device_name(0), '| capability', torch.cuda.get_device_capability(0))
    x = torch.randn(2048, 2048, device='cuda'); print('matmul ok', float((x @ x).sum()) == float((x @ x).sum()))
EOF"
    in_env "$SANDBED_ENV" "python -c \"import go2_env, go2_train, go2_fsm_sim; print('project modules import ok')\""
    in_env "$SANDBED_ENV" "pip check"
    in_env "$SANDBED_ENV" "python - <<'EOF'
import re, importlib.metadata as md
from pathlib import Path
norm = lambda n: re.sub(r'[-_.]+', '-', n).lower()
p = Path('env/requirements-lock.txt')
if not p.exists():
    print('no env/requirements-lock.txt'); raise SystemExit
now = {norm(d.metadata['Name']): d.version for d in md.distributions()}
print('--- versions that differ from env/requirements-lock.txt (the Windows lock):')
diff = 0
for line in p.read_text().splitlines():
    if '==' not in line or ' @ ' in line or line.startswith('#'):
        continue
    n, v = (x.strip() for x in line.split('==', 1))
    m = now.get(norm(n), 'MISSING')
    if m != v:
        diff += 1; print(f'{norm(n):24s} lock {v:20s} now {m}')
print(f'{diff} difference(s); torch +cu suffix, msvc-runtime and colorama are expected')
EOF"
    in_env "$SANDBED_ENV" "python - <<'EOF'
from importlib.util import find_spec
from pathlib import Path
d = Path(find_spec('genesis').origin).parent / 'assets' / 'urdf' / 'plane'
bad = [p.name for p in sorted(d.glob('*.urdf')) if 'version=\"0.0\"' in p.read_text().splitlines()[0]]
print('genesis plane URDF patch:', 'NEEDED -> run python fix_plane_urdf.py ' + str(bad) if bad else 'applied (ok)')
EOF"
    if [ -z "$QUICK" ]; then
      in_env "$SANDBED_ENV" "python - <<'EOF'
import time, genesis as gs
t = time.time(); gs.init(backend=gs.gpu, logging_level='warning')
scene = gs.Scene(show_viewer=False); scene.add_entity(gs.morphs.Plane()); scene.build()
for _ in range(10): scene.step()
print(f'genesis gpu scene ok ({time.time() - t:.0f} s)')
EOF" 600
      in_env "$SANDBED_ENV" "PYOPENGL_PLATFORM=egl python - <<'EOF'
import genesis as gs
gs.init(backend=gs.gpu, logging_level='warning')
scene = gs.Scene(show_viewer=False); scene.add_entity(gs.morphs.Plane())
cam = scene.add_camera(res=(64, 64), pos=(2, 2, 2), lookat=(0, 0, 0), GUI=False)
scene.build(); out = cam.render()
print('genesis EGL camera render ok', getattr(out[0] if isinstance(out, tuple) else out, 'shape', type(out)))
EOF" 600
      if [ -n "$EXP" ]; then
        in_env "$SANDBED_ENV" "if python go2_fsm_sim.py --help 2>&1 | grep -q -- '--log'; then a=\"--scenario stand --log logs/deploy/genesis_sysreport.npz\"; else a=\"--scenario full --out /tmp/go2_sysreport_fsm.png\"; fi
echo \"(fsm_sim args: \$a)\"; python go2_fsm_sim.py -e '$EXP' \$a --gpu 2>&1 | grep -vE 'it/s|%\\|' | tail -n 25" 900
        in_env "$SANDBED_ENV" "python go2_train.py -e sysreport_smoke -b 256 --max-iterations 2 2>&1 | tail -n 25 && rm -rf logs/sysreport_smoke" 900
      fi
    fi

    sec "5. Deploy env: $DEPLOY_ENV"
    run "git -C '$DDS_SRC' log -1 --format='cyclonedds C lib: %h %cd %s'; git -C '$DDS_SRC' branch --show-current"
    in_env "$DEPLOY_ENV" "which python; echo CYCLONEDDS_HOME=\$CYCLONEDDS_HOME; ls \$CYCLONEDDS_HOME/lib 2>&1 | grep -E 'ddsc|cyclonedds' | head; python - <<'EOF'
import sys, importlib.metadata as md
print('python', sys.version.split()[0], sys.executable)
for p in ('cyclonedds', 'unitree_sdk2py', 'mujoco', 'torch', 'numpy', 'pygame', 'opencv-python', 'matplotlib'):
    try: print(f'{p:14s}', md.version(p))
    except Exception: print(f'{p:14s}', '-')
import cyclonedds, unitree_sdk2py
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
print('sdk import ok:', unitree_sdk2py.__file__)
EOF"
    for gl in egl glfw osmesa; do
      in_env "$DEPLOY_ENV" "MUJOCO_GL=$gl python -c \"
import mujoco
m = mujoco.MjModel.from_xml_string('<mujoco><worldbody><light pos=\\\"0 0 2\\\"/><geom type=\\\"sphere\\\" size=\\\".1\\\"/></worldbody></mujoco>')
r = mujoco.Renderer(m, 64, 64); d = mujoco.MjData(m); mujoco.mj_forward(m, d); r.update_scene(d)
print('MUJOCO_GL=$gl render ok', r.render().shape)\" 2>&1 | tail -n 2" 60
    done
    in_env "$DEPLOY_ENV" "SDL_VIDEODRIVER=dummy python -c \"
import pygame; pygame.init(); pygame.joystick.init(); n = pygame.joystick.get_count(); print('gamepads seen by pygame:', n)
[print(' ', pygame.joystick.Joystick(i).get_name()) for i in range(n)]\" 2>&1 | grep -v 'Hello from\\|pygame 2\\|ALSA'" 30
  fi

  sec "6. Project folder"
  run "git -C '$PROJ' log -1 --format='%h %cd %s'; git -C '$PROJ' branch --show-current; git -C '$PROJ' remote -v | head -1; echo '--- uncommitted / untracked:'; git -C '$PROJ' status --short | head -30"
  run "cd '$PROJ' && for f in go2_env.py go2_train.py go2_export.py go2_metrics.py go2_fsm.py go2_fsm_sim.py go2_deploy_sdk2.py go2_commands.py go2_probe.py go2_analyze.py go2_inputs.py go2_log.py go2_kinematics.py go2_mujoco_headless.py test_go2_deploy.py run_sim_gate.sh run_all.sh scene_flat.xml README_S2R.md fix_plane_urdf.py; do [ -e \$f ] || echo MISSING \$f; done; echo 'file check done'; ls -l *.sh | awk '{print \$1, \$NF}'"
  run "ls -la '$PROJ'"
  run "cd '$PROJ' && file *.sh *.py 2>/dev/null | grep -E 'CRLF|script' ; echo '--- files with Windows line endings (CR):'; grep -lI \$'\\r' *.py *.sh *.xml *.json *.md 2>/dev/null"
  run "cd '$PROJ' && echo '--- Windows paths in code:'; grep -nE '(^|[^a-zA-Z])[a-zA-Z]:[\\\\/]' *.py *.sh *.json 2>/dev/null | head -20"
  run "cd '$PROJ' && ls logs/ && ls -la logs/${EXP:-NONE}/ logs/${EXP:-NONE}/exported/ 2>&1 | head -40"
  run "cd '$PROJ' && head -n 25 run_sim_gate.sh"
  run "cd '$PROJ' && grep -nE 'include|file=' scene_flat.xml 2>/dev/null | head"
  run "cd '$PROJ' && du -sh logs nightly 2>/dev/null; echo '--- last overnight queue:'; ls -d nightly/*/ 2>/dev/null | tail -1 | xargs -r -I{} cat {}summary.txt"

  sec "7. unitree_mujoco and unitree_sdk2_python"
  run "git -C '$UM' log -1 --format='%h %cd %s'; git -C '$UM' status --short | head; git -C '$SDK' log -1 --format='%h %cd %s'"
  run "grep -nE '^[A-Z_]+ *=' '$UM/simulate_python/config.py'"
  run "ls '$UM/unitree_robots/go2/'"

  sec "8. Network (for DDS now, the robot later)"
  run "ip -br link | sed -E 's/([0-9a-f]{2}:){5}[0-9a-f]{2}/xx:xx:xx:xx:xx:xx/g'; ip -br addr"
  run "ip link show lo | head -1; ip link show lo | grep -q MULTICAST && echo 'lo multicast: on' || echo 'lo multicast: OFF (CycloneDDS on lo may not discover peers)'; ip route"
  run "nmcli -t -f DEVICE,TYPE,STATE,CONNECTION device 2>/dev/null"
  run "systemctl is-active ufw firewalld 2>/dev/null"

  sec "9. Real-time priority and limits"
  run "id; ulimit -r; ulimit -l"
  run "grep -hvE '^#|^\$' /etc/security/limits.conf /etc/security/limits.d/* 2>/dev/null"
  run "chrt -f 50 true && echo 'SCHED_FIFO 50 allowed for this user' || echo 'SCHED_FIFO refused (no rtprio limit)'"
  run "cat /sys/kernel/realtime 2>/dev/null; uname -v"

  sec "10. Workflow tools"
  run "for t in git tmux systemd-inhibit nvtop cmake gcc; do printf '%-16s %s\\n' \$t \"\$(command -v \$t || echo MISSING)\"; done"

  sec "10b. Gamepad devices"
  run "ls -l /dev/input/js* /dev/input/by-id/ 2>&1 | head -20"

  if [ -n "$CONDA_BASE" ]; then
    sec "11. Smoke tests, deploy side"
    if [ -n "$EXP" ] && [ -f "$PROJ/test_go2_deploy.py" ]; then
      in_env "$DEPLOY_ENV" "python test_go2_deploy.py --policy logs/$EXP/exported --dds 2>&1 | grep -E '^(ok|FAIL)|check|Error' | tail -n 45" 300
    fi
    [ -f "$PROJ/go2_mujoco_headless.py" ] && [ -f "$PROJ/go2_probe.py" ] || echo "skip sim+probe test: go2_mujoco_headless.py / go2_probe.py not in $PROJ"
    [ -f "$PROJ/go2_mujoco_headless.py" ] && [ -f "$PROJ/go2_probe.py" ] && in_env "$DEPLOY_ENV" "python go2_mujoco_headless.py --um '$UM' --dt 0.002 --duration 15 > /tmp/go2_sysreport_sim.txt 2>&1 &
sleep 7; python go2_probe.py --sim --seconds 4 2>&1 | tail -n 30; wait; echo '--- simulator:'; tail -n 6 /tmp/go2_sysreport_sim.txt" 90
  fi

  sec "done"
  echo "report: $OUT"
}

main 2>&1 | tee "$OUT"