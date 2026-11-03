"""System Lagging が続いたブートの状態を残す (GS450h 追加)。

背景: 11-02 route 122 で、起動 54 秒後から selfdrivedLagging (TAKE CONTROL IMMEDIATELY / System Lagging) が出続けた。
card / controlsd / selfdrived は 3 つとも core 4 に固定されていて平常でも 94% 使っている。そのブートだけ big core 側と
GPU が約 1.4 倍遅く、core 4 が溢れた。クロック (CPU 1.69GHz / GPU 675MHz) も DDR の実測も平常と同じで、再起動で治った。
原因を追うにはカーネルログと SoC の状態が要るが、journald は永続化されていないので再起動で消える (11-02 はこれで取り逃した)。

取るもの:
  - lag: そのブートで selfdrivedLagging が LAG_ONSETS_BEFORE_SNAPSHOT 回出たか、合計 LAG_SECONDS_BEFORE_SNAPSHOT 秒続いたら 1 回
  - baseline: onroad になって BASELINE_AFTER_ONROAD 秒たったら 1 回 (比べる相手。11-02 は平常時の値が無くて判定できなかった)
中身: dmesg / CPU・GPU・DDR・L3 の実クロック / LLC slice / 熱 / 割り込み / メモリ / top -H / core 4 のプロセスの CPU 使用率 (5 秒)
出力: `/data/community/lag_snapshots/<日時>-up<uptime>s-<kind>/`、新しい方から KEEP 個だけ残す。swaglog に `lag snapshot` イベント。

⚠ import 時は標準ライブラリだけ (CI が依存なしで挙動テストを回すため)。ログ出力は呼び出し側が渡す。
⚠ hardwared を落とさないこと: 例外はすべて握る。書き出しは別スレッドで、小 core (0-3) に寄せて core 4 を邪魔しない。
"""
import datetime
import os
import shutil
import subprocess
import threading
import time
from collections.abc import Callable

SNAPSHOT_DIR = '/data/community/lag_snapshots'
REALDATA_DIR = '/data/media/0/realdata'
VERSION_PARAM = '/data/params/d/UpdaterCurrentDescription'
LAG_ONSETS_BEFORE_SNAPSHOT = 3     # selfdrivedLagging の立ち上がりがこの回数に達したら取る
LAG_SECONDS_BEFORE_SNAPSHOT = 10.  # [s] 立ち上がり 1 回でも出っぱなしならこの秒数で取る
BASELINE_AFTER_ONROAD = 180.       # [s] onroad になってから平常時の値を取るまで
KEEP = 8                           # 残すスナップショットの数 (/data は空きが少ない。1 個 数百 KB)
CPU_SAMPLE_SECONDS = 5.
LITTLE_CORES = {0, 1, 2, 3}

# core 4 / 7 に固定されたプロセス (cmdline の一部 → 表示名)
WATCH_PROCS = {
  'selfdrive.car.card': 'card', 'controls.controlsd': 'controlsd', 'selfdrived.selfdrived': 'selfdrived',
  'modeld.modeld': 'modeld', 'modeld.dmonitoringmodeld': 'dmonitoringmodeld',
}

# root でしか読めないもの (debugfs の実測クロック・cpuinfo_cur_freq) も含めて 1 回の sudo で読む
SYSFS_SCRIPT = r'''
for p in 0 4; do d=/sys/devices/system/cpu/cpufreq/policy$p
  echo "policy$p gov=$(cat $d/scaling_governor) max=$(cat $d/scaling_max_freq) cur=$(cat $d/scaling_cur_freq) hw=$(cat $d/cpuinfo_cur_freq)"
  echo "policy$p time_in_state: $(tr '\n' ' ' < $d/stats/time_in_state)"
done
echo "cpus_online=$(cat /sys/devices/system/cpu/online)"
for c in measure_only_bimc_clk l3_clk gpu_cc_gx_gfx3d_clk; do
  echo "$c measure: $(for i in 1 2 3; do cat /sys/kernel/debug/clk/$c/clk_measure; done | tr '\n' ' ')"
done
K=/sys/class/kgsl/kgsl-3d0
for f in gpuclk max_gpuclk min_pwrlevel max_pwrlevel thermal_pwrlevel gpu_llc_slice_enable gpuhtw_llc_slice_enable gpubusy force_clk_on force_bus_on; do
  echo "kgsl $f=$(cat $K/$f)"
done
for d in /sys/class/devfreq/*; do
  echo "devfreq $(basename $d) gov=$(cat $d/governor) cur=$(cat $d/cur_freq) min=$(cat $d/min_freq) max=$(cat $d/max_freq)"
done
for z in /sys/class/thermal/thermal_zone*; do echo "thermal $(cat $z/type)=$(cat $z/temp)"; done
for d in /sys/class/thermal/cooling_device*; do s=$(cat $d/cur_state); [ "$s" != "0" ] && echo "cooling $(cat $d/type)=$s/$(cat $d/max_state)"; done
echo done
'''


def _run(cmd: list[str], timeout: float = 20.) -> str:
  try:
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return r.stdout + (f'\n[stderr]\n{r.stderr}' if r.stderr.strip() else '')
  except Exception as e:
    return f'[{" ".join(cmd[:3])} failed: {e!r}]'


def _read(path: str) -> str:
  try:
    with open(path) as f:
      return f.read()
  except Exception as e:
    return f'[{path}: {e!r}]'


def _jiffies(pid: int) -> int | None:
  try:
    with open(f'/proc/{pid}/stat') as f:
      fields = f.read().rsplit(')', 1)[1].split()
    return int(fields[11]) + int(fields[12])  # utime + stime
  except Exception:
    return None


def _cpu_times() -> dict[str, list[int]]:
  out = {}
  for line in _read('/proc/stat').splitlines():
    if line.startswith('cpu') and line[3:4].isdigit():
      name, *vals = line.split()
      out[name] = [int(v) for v in vals]
  return out


def sample_cpu(seconds: float = CPU_SAMPLE_SECONDS) -> str:
  """core 4 / 7 に固定されたプロセスの CPU 使用率と、core ごとの使用率を `seconds` 秒で測る。"""
  pids: dict[str, int] = {}
  for p in os.listdir('/proc'):
    if not p.isdigit():
      continue
    cmd = _read(f'/proc/{p}/cmdline')
    for key, name in WATCH_PROCS.items():
      if key in cmd and name not in pids:
        pids[name] = int(p)
  j0 = {n: _jiffies(p) for n, p in pids.items()}
  c0 = _cpu_times()
  time.sleep(seconds)
  j1 = {n: _jiffies(p) for n, p in pids.items()}
  c1 = _cpu_times()
  hz = os.sysconf('SC_CLK_TCK') if hasattr(os, 'sysconf') else 100
  lines = [f'per-process cpu over {seconds:.0f}s (100% = 1 core):']
  for n in pids:
    a, b = j0[n], j1[n]
    lines.append(f'  {n:18s} ' + (f'{(b - a) / hz / seconds * 100:5.1f}%' if a is not None and b is not None else 'gone'))
  lines.append('per-core busy:')
  for name in sorted(c1, key=lambda s: int(s[3:])):
    if name not in c0:
      continue
    d = [y - x for x, y in zip(c0[name], c1[name], strict=False)]
    total = sum(d)
    idle = d[3] + (d[4] if len(d) > 4 else 0)
    lines.append(f'  {name}: {(total - idle) / total * 100 if total else 0:5.1f}%')
  return '\n'.join(lines)


def _latest_route(realdata: str = REALDATA_DIR) -> str:
  """いま記録中の segment (rlog と突き合わせるため)。

  ⚠ 更新時刻では選ばない: 起動直後は時計が 07-28 に化けていて、新しい route の方が古い時刻になる (11-02 実機で外れた)。
  名前 `<route 番号 hex>--<id>--<seg>` の route 番号と seg 番号で選ぶ。
  """
  def key(d: str) -> tuple[int, int]:
    route, _, seg = d.split('--')
    return int(route, 16), int(seg)

  try:
    segs = []
    for d in os.listdir(realdata):
      try:
        segs.append((key(d), d))
      except ValueError:
        pass
    return max(segs)[1] if segs else '?'
  except Exception:
    return '?'


def _pin_to_little_cores() -> None:
  # Linux の sched_setaffinity(0) は呼んだスレッドだけに効く。core 4 (card/controlsd/selfdrived) を邪魔しない
  try:
    os.sched_setaffinity(0, LITTLE_CORES)
  except Exception:
    pass


def capture(kind: str, out_dir: str, summary: dict) -> str:
  """スナップショットを 1 個書いて、そのディレクトリを返す。"""
  _pin_to_little_cores()
  uptime = _read('/proc/uptime').split()[0].split('.')[0] if os.path.exists('/proc/uptime') else '?'
  path = os.path.join(out_dir, f'{datetime.datetime.now():%Y%m%d-%H%M%S}-up{uptime}s-{kind}')
  os.makedirs(path, exist_ok=True)
  little = ['taskset', '-c', '0-3', 'nice', '-n', '10']

  def write(name: str, text: str) -> None:
    with open(os.path.join(path, name), 'w') as f:
      f.write(text)

  # 先に CPU を測る (後の dmesg / top の負荷を混ぜない)
  cpu = sample_cpu()
  write('summary.txt', '\n'.join(f'{k}: {v}' for k, v in summary.items()) +
        f'\nuptime: {uptime}s\nroute: {_latest_route()}\nversion: {_read(VERSION_PARAM).strip()}\n\n{cpu}\n')
  write('sysfs.txt', _run(little + ['sudo', 'sh', '-c', SYSFS_SCRIPT], timeout=30))
  # c4 の dmesg は wlan のビーコン処理でほぼ埋まるので落とす (save_dmesg_snapshot と同じ)
  dmesg = _run(little + ['dmesg', '-T'], timeout=20)
  write('dmesg.txt', '\n'.join(ln for ln in dmesg.splitlines() if 'wlan:' not in ln)[-400000:])
  write('top.txt', _run(little + ['top', '-b', '-n', '1', '-H', '-o', '%CPU', '-w', '200'])[:60000] +
        '\n\n' + _run(little + ['ps', '-eLo', 'psr,rtprio,pcpu,pid,tid,comm', '--sort=-pcpu'])[:40000])
  write('proc.txt', '\n'.join(f'==== {p}\n{_read(p)}' for p in
                              ('/proc/loadavg', '/proc/meminfo', '/proc/vmstat', '/proc/interrupts', '/proc/softirqs')))
  return path


def prune(out_dir: str, keep: int = KEEP) -> None:
  # ⚠ 名前 (日時) では並べない: 起動直後は RTC 既定値で 07-28 に化けるので、新しいものほど古い名前になりうる
  try:
    dirs = sorted((d for d in os.listdir(out_dir) if os.path.isdir(os.path.join(out_dir, d))),
                  key=lambda d: os.path.getmtime(os.path.join(out_dir, d)))
  except FileNotFoundError:
    return
  for d in dirs[:-keep] if keep > 0 else dirs:
    shutil.rmtree(os.path.join(out_dir, d), ignore_errors=True)


class LagSnapshot:
  """hardwared のループから毎周 `update()` を呼ぶ。取るかどうかの判定だけで、書き出しは別スレッド。"""

  def __init__(self, log_event: Callable[..., None], out_dir: str = SNAPSHOT_DIR,
               capture_fn: Callable[[str, str, dict], str] = capture, background: bool = True):
    self.log_event = log_event
    self.out_dir = out_dir
    self.capture_fn = capture_fn
    self.background = background
    self.lag_onsets = 0
    self.lag_seconds = 0.
    self.lagging = False
    self.lag_taken = False
    self.baseline_taken = False
    self._last_t: float | None = None

  def update(self, now: float, onroad_since: float | None, lagging: bool) -> None:
    dt = 0. if self._last_t is None else max(0., now - self._last_t)
    self._last_t = now
    if onroad_since is None:
      self.lagging = False
      return
    if lagging:
      if not self.lagging:
        self.lag_onsets += 1
      self.lag_seconds += dt if self.lagging else 0.
    self.lagging = lagging

    if not self.lag_taken and (self.lag_onsets >= LAG_ONSETS_BEFORE_SNAPSHOT or self.lag_seconds >= LAG_SECONDS_BEFORE_SNAPSHOT):
      self.lag_taken = True
      self._start('lag', now - onroad_since)
    elif not self.baseline_taken and now - onroad_since >= BASELINE_AFTER_ONROAD:
      self.baseline_taken = True
      self._start('baseline', now - onroad_since)

  def _start(self, kind: str, onroad_for: float) -> None:
    summary = {'kind': kind, 'onroad_for': f'{onroad_for:.0f}s', 'lag_onsets': self.lag_onsets,
               'lag_seconds': f'{self.lag_seconds:.1f}'}
    if self.background:
      threading.Thread(target=self._run, args=(kind, summary), name=f'lag_snapshot_{kind}', daemon=True).start()
    else:
      self._run(kind, summary)

  def _run(self, kind: str, summary: dict) -> None:
    try:
      os.makedirs(self.out_dir, exist_ok=True)
      path = self.capture_fn(kind, self.out_dir, summary)
      prune(self.out_dir)
      self.log_event('lag snapshot', path=path, **summary)
    except Exception as e:
      try:
        self.log_event('lag snapshot failed', kind=kind, error=repr(e))
      except Exception:
        pass
