"""hardwared のループが止まった瞬間を捕まえる watchdog (GS450h 追加)。

背景: 09-21 / 09-25 に hardwared (deviceState の publisher) だけが ~7.5s 止まり、selfdrived が
`commIssue` で強制解除した (計 4 回)。他のサービスは止まっておらず、hardwared はその間 CPU を使っていない
= カーネル内の何かで待たされている。rlog / swaglog からは「どの呼び出しか」が分からないので、落ちる側で残す。

取るもの (止まったときだけ):
  1. 全スレッドの Python スタック — `faulthandler.dump_traceback_later` の C スレッドが GIL 無しで書く
     (GIL を握ったまま止まっていても出る)。ループの `beat()` が約 1 秒ごとに装填し直し、
     `threshold` 秒 `beat()` が来なければ発火する。
  2. 止まったスレッドがカーネルのどこで待っているか — 監視スレッドが `/proc/self/task/<tid>/`
     の stat (状態) / wchan / syscall を読み、syscall の fd が指すファイルを readlink で引く。
     監視スレッド自体が遅れたら (= GIL を握られていた) その秒数も残す。
  3. 止まっていた秒数と dmesg (再起動で消える) — ループが戻った後に監視スレッドが書く。
出力: crash 置き場 (`/data/community/crashes`) に `hwstall-<ts>.log` + `dmesg-<ts>-hwstall.log`、
      swaglog に `hardwared stall` イベント。1 秒以上のループの空きは `hardwared loop gap` (ファイル無し)。

⚠ import 時は標準ライブラリだけ (CI が依存なしで挙動テストを回すため)。ログ出力と書き出しは呼び出し側が渡す。
⚠ hardwared を落とさないこと: 監視側の例外はすべて握って続行する。止まっている間はディスクに書かない
   (止まった原因がディスクなら巻き込まれるため。書くのは戻った後)。
"""
import datetime
import faulthandler
import os
import platform
import tempfile
import threading
import time
from collections.abc import Callable

# aarch64 (c4) の syscall 番号。読めた番号を名前にするだけ (無い番号は数字のまま出す)。
# ⚠ 番号はアーキテクチャごとに違う (x86_64 の read は 0) ⇒ aarch64 以外では名前も fd も引かない。
AARCH64 = platform.machine() in ('aarch64', 'arm64')
SYSCALL_NAMES_AARCH64 = {
  17: 'getcwd', 22: 'epoll_pwait', 23: 'dup', 25: 'fcntl', 29: 'ioctl', 32: 'flock', 34: 'mkdirat', 35: 'unlinkat',
  38: 'renameat', 43: 'statfs', 44: 'fstatfs', 46: 'ftruncate', 48: 'faccessat', 56: 'openat', 57: 'close',
  61: 'getdents64', 62: 'lseek', 63: 'read', 64: 'write', 65: 'readv', 66: 'writev', 67: 'pread64', 68: 'pwrite64',
  72: 'pselect6', 73: 'ppoll', 78: 'readlinkat', 79: 'newfstatat', 80: 'fstat', 82: 'fsync', 83: 'fdatasync',
  98: 'futex', 101: 'nanosleep', 115: 'clock_nanosleep', 124: 'sched_yield', 203: 'connect', 206: 'sendto',
  207: 'recvfrom', 211: 'sendmsg', 212: 'recvmsg', 220: 'clone', 221: 'execve', 260: 'wait4', 276: 'renameat2',
  291: 'statx',
}
# 第 1 引数が fd の syscall (fd → /proc/self/fd で対象ファイルが分かる)
FD_SYSCALLS_AARCH64 = {22, 23, 25, 29, 32, 44, 46, 57, 61, 62, 63, 64, 65, 66, 67, 68, 80, 82, 83, 203, 206, 207, 211, 212}
SYSCALL_NAMES = SYSCALL_NAMES_AARCH64 if AARCH64 else {}
FD_SYSCALLS = FD_SYSCALLS_AARCH64 if AARCH64 else set()
# 監視スレッドの起床がこれ以上遅れたら「GIL を握られていた」とみなす (poll は 0.25s、平常時の遅れは数 ms)
GIL_LATE = 0.5


def _read(path: str) -> str:
  try:
    with open(path) as f:
      return f.read().strip()
  except Exception:
    return '?'


def sample_task(tid: int, proc_root: str = '/proc/self/task') -> dict:
  """1 スレッドの「今カーネルのどこにいるか」。/proc が無い環境 (PC) では '?' が並ぶだけ。"""
  base = f'{proc_root}/{tid}'
  stat = _read(f'{base}/stat')
  state = '?'
  if ')' in stat:
    rest = stat.rsplit(')', 1)[1].split()
    state = rest[0] if rest else '?'
  syscall = _read(f'{base}/syscall')
  s = {'tid': tid, 'state': state, 'wchan': _read(f'{base}/wchan'), 'syscall': syscall}
  parts = syscall.split()
  if parts and parts[0].lstrip('-').isdigit():
    nr = int(parts[0])
    s['syscall_name'] = SYSCALL_NAMES.get(nr, str(nr))
    if nr in FD_SYSCALLS and len(parts) > 1:
      try:
        fd = int(parts[1], 16)
        s['fd'] = fd
        s['fd_target'] = os.readlink(f'/proc/self/fd/{fd}')
      except Exception:
        pass
  return s


def _fmt_sample(s: dict) -> str:
  out = f"state={s.get('state')} wchan={s.get('wchan')} syscall={s.get('syscall_name', s.get('syscall'))}"
  if 'fd_target' in s:
    out += f" fd={s['fd']} -> {s['fd_target']}"
  return out


class StallWatchdog:
  def __init__(self, name: str, log_event: Callable[..., None] | None = None,
               write_report: Callable[[str], object] | None = None, save_dmesg: Callable[[], object] | None = None,
               threshold: float = 3.0, rearm_interval: float = 1.0, poll: float = 0.25,
               gap_log_threshold: float = 1.0, max_reports: int = 20, max_samples: int = 40):
    self.name = name
    self.log_event = log_event or (lambda *a, **k: None)
    self.write_report = write_report or (lambda text: None)
    self.save_dmesg = save_dmesg or (lambda: None)
    self.threshold = threshold
    self.rearm_interval = rearm_interval
    self.poll = poll
    self.gap_log_threshold = gap_log_threshold
    self.max_reports = max_reports
    self.max_samples = max_samples

    self.reports = 0
    self._tid: int | None = None
    self._last_beat: float | None = None
    self._last_arm = 0.
    self._gaps: list[float] = []
    self._armed = False
    self._thread: threading.Thread | None = None
    self._stop = threading.Event()
    self._tb_file = self._open_tb_file()

  @staticmethod
  def _open_tb_file():
    # faulthandler はここへ書く。メモリ上に置く (止まった原因がディスクなら、ディスクへの書き込みは巻き込まれる)。
    # ⚠ 開けなくても例外にしない (hardwared が起きないと deviceState が出ず engage できない)。Python スタックだけ諦める
    try:
      if hasattr(os, 'memfd_create'):
        return open(os.memfd_create('hwstall_tb'), 'w+b', buffering=0)
      return tempfile.TemporaryFile(buffering=0)
    except Exception:
      return None

  def _tb_read_and_clear(self) -> str:
    if self._tb_file is None:
      return '(faulthandler の出力先を開けなかったので Python スタックは無し)'
    try:
      fd = self._tb_file.fileno()
      size = os.fstat(fd).st_size
      data = os.pread(fd, size, 0) if hasattr(os, 'pread') else self._seek_read()
      os.ftruncate(fd, 0)
      os.lseek(fd, 0, os.SEEK_SET)
      return data.decode('utf-8', 'replace')
    except Exception as e:
      return f'(faulthandler 出力を読めなかった: {e!r})'

  def _seek_read(self) -> bytes:
    self._tb_file.seek(0)
    return self._tb_file.read()

  def start(self) -> None:
    if self._thread is None:
      self._thread = threading.Thread(target=self._monitor, name=f'{self.name}_stall_watchdog', daemon=True)
      self._thread.start()

  def stop(self) -> None:
    self._stop.set()
    if self._armed:
      faulthandler.cancel_dump_traceback_later()
      self._armed = False

  def beat(self) -> None:
    """ループの先頭で毎回呼ぶ。⚠ ループの一部なので軽く保つ (時刻の代入と、約 1 秒ごとの装填だけ)。"""
    now = time.monotonic()
    if self._tid is None:
      self._tid = threading.get_native_id()
    if self._last_beat is not None and now - self._last_beat >= self.gap_log_threshold:
      self._gaps.append(now - self._last_beat)
    self._last_beat = now
    if now - self._last_arm >= self.rearm_interval:
      self._last_arm = now
      if self.reports < self.max_reports and self._tb_file is not None:
        try:
          faulthandler.dump_traceback_later(self.threshold, repeat=False, file=self._tb_file, exit=False)
          self._armed = True
        except Exception:
          pass

  # ---- 監視スレッド ----

  def _monitor(self) -> None:
    stall: dict | None = None
    prev = time.monotonic()
    while not self._stop.wait(self.poll):
      try:
        now = time.monotonic()
        late = now - prev - self.poll  # 監視スレッドが起きるのが遅れた秒数 (大きい = GIL を握られていた)
        prev = now
        gaps = []
        while self._gaps:
          gaps.append(self._gaps.pop(0))
        for gap in gaps:
          if gap < self.threshold:  # threshold 以上は下の stall 側で出す
            self.log_event(f'{self.name} loop gap', gap=round(gap, 3))
        last = self._last_beat
        if last is None:
          continue
        big = [g for g in gaps if g >= self.threshold]
        if stall is None and big:
          # 止まっている間に監視スレッドも動けなかった (= GIL を握ったまま止まっていた)。/proc のサンプルは無いが、
          # faulthandler の C スレッドは GIL 無しでスタックを書いているので、それを回収してレポートにする
          st = self._begin_stall(last - big[-1], now, sample=False)
          st['late_max'], st['monitor_blocked'] = late, True
          self._finish_stall(st, big[-1])
          continue
        age = now - last
        if age >= self.threshold:
          if stall is None:
            stall = self._begin_stall(last, now)
          stall['late_max'] = max(stall['late_max'], late)
          # 起床が大きく遅れた = その間 GIL を握られていた (C 呼び出しから戻った直後、beat より先に GIL が回ってきた形)
          stall['monitor_blocked'] |= late >= GIL_LATE
          if len(stall['samples']) < self.max_samples and self._tid is not None:
            stall['samples'].append((now - last, sample_task(self._tid)))
        elif stall is not None and last > stall['beat']:
          self._finish_stall(stall, max(gaps) if gaps else last - stall['beat'])
          stall = None
        elif age < 1.0 and self._tb_size() > 0:
          # threshold 未満の短い停止で faulthandler だけ発火した残り。次の本物のレポートに混ざらないよう捨てる
          self._tb_read_and_clear()
      except Exception:
        pass

  def _tb_size(self) -> int:
    if self._tb_file is None:
      return 0
    try:
      return os.fstat(self._tb_file.fileno()).st_size
    except Exception:
      return 0

  def _begin_stall(self, last: float, now: float, sample: bool = True) -> dict:
    """last = 止まる直前の beat (monotonic)。sample=False は「止まっている間に監視できなかった」事後の組み立て。"""
    names = {t.native_id: t.name for t in threading.enumerate()}
    tasks = []
    if sample:
      try:
        for tid in sorted(int(t) for t in os.listdir('/proc/self/task')):
          s = sample_task(tid)
          s['name'] = names.get(tid, '?')
          tasks.append(s)
      except Exception:
        pass
      first = sample_task(self._tid) if self._tid is not None else {}
      self.log_event(f'{self.name} stall detected', error=True, age=round(now - last, 2),
                     thread=names.get(self._tid, '?'), **{k: v for k, v in first.items() if k != 'tid'})
    # 壁時計は swaglog / crash ログのファイル名と突き合わせるためだけに使う
    wall = datetime.datetime.now() - datetime.timedelta(seconds=now - last)
    return {'beat': last, 'wall': wall, 'late_max': 0., 'samples': [], 'monitor_blocked': False,
            'tasks': tasks, 'thread_name': names.get(self._tid, '?')}

  def _finish_stall(self, stall: dict, gap: float) -> None:
    tb = self._tb_read_and_clear()
    first = stall['samples'][0][1] if stall['samples'] else {}
    if self.reports >= self.max_reports:
      self.log_event(f'{self.name} stall', error=True, gap=round(gap, 2), report=None, note='max_reports')
      return
    self.reports += 1
    lines = [
      f'{self.name} stall report',
      f"thread: {stall['thread_name']} tid={self._tid}",
      f"gap: {gap:.2f} s (最後の beat = {stall['wall']:%Y-%m-%d %H:%M:%S.%f})",
      f"監視スレッドの起床遅れ max: {stall['late_max']:.2f} s ({GIL_LATE} s 以上 = GIL を握られていた)",
    ]
    if stall['monitor_blocked']:
      lines.append('⚠ 止まっている間は監視スレッドも動けなかった = GIL を握ったまま止まっていた')
      lines.append('  (カーネル側のサンプルは止まった後に取ったもの = 当てにしない。Python スタックを見る)')
    lines += [
      '',
      '--- 止まったスレッドのカーネル側 (最後の beat からの秒数) ---',
    ]
    # 同じ場所で待ち続けている間のサンプルは 1 行にまとめる (待ち場所が変わった瞬間だけ行が増える)
    runs: list[list] = []
    for age, s in stall['samples']:
      text = _fmt_sample(s)
      if runs and runs[-1][2] == text:
        runs[-1][1], runs[-1][3] = age, runs[-1][3] + 1
      else:
        runs.append([age, age, text, 1])
    lines += [f'  +{a0:5.2f}s .. +{a1:5.2f}s (x{n}) {text}' for a0, a1, text, n in runs]
    lines += ['', '--- 検出時点の全スレッド ---']
    lines += [f"  tid={s['tid']} name={s.get('name')} {_fmt_sample(s)}" for s in stall['tasks']]
    lines += ['', '--- faulthandler (全スレッドの Python スタック、GIL 無しで取得) ---', tb or '(発火していない)']
    path = None
    try:
      path = self.write_report('\n'.join(lines))
    except Exception:
      pass
    try:
      self.save_dmesg()
    except Exception:
      pass
    self.log_event(f'{self.name} stall', error=True, gap=round(gap, 2), report=str(path) if path else None,
                   late_max=round(stall['late_max'], 2), gil_held=stall['monitor_blocked'],
                   **{k: v for k, v in first.items() if k != 'tid'})
