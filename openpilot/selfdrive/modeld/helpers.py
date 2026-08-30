import collections
import datetime
import fcntl
import io
import json
import os
import pickle
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from openpilot.common.file_chunker import get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id
from openpilot.common.swaglog import cloudlog

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
CHESTNUT_POWERED_VOLTAGE = 5000
CHESTNUT_PCIE_READY = 0x78

# 落ちたときのスナップショット置き場 (crash ログと同じところ)。テストから差し替えられるよう定数にしてある。
CRASH_DIR = Path('/data/community/crashes')


def get_tg_input_devices(process_name: str, chestnut: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not chestnut else 'chestnut']

def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      f.readinto(pb)
      yield pb
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

def chestnut_device_path() -> Path | None:
  """chestnut (eGPU) の sysfs デバイスディレクトリ。見つからなければ None。"""
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return d
    except Exception:
      pass
  return None


def chestnut_present() -> bool:
  return chestnut_device_path() is not None


# _IO('U', 20) — USB デバイスをリセットする ioctl。sysfs の authorized と違い devnode に対して
# 効くので、comma ユーザー (openpilot の実行ユーザー) のままでも叩ける。
USBDEVFS_RESET = ord('U') << 8 | 20


def _chestnut_devnode(d: Path) -> Path | None:
  try:
    return Path(f"/dev/bus/usb/{int((d / 'busnum').read_text()):03d}/{int((d / 'devnum').read_text()):03d}")
  except Exception:
    return None


def save_dmesg_snapshot(tag: str = "") -> Path | None:
  """クラッシュ直前のカーネルログを crash ログの隣に残す (GS450h 追加、08-30)。

  ⚠⚠ journald は永続化されていない (`/var/log/journal` 無し) ので、**再起動すると dmesg は消える**。
  chestnut のハングは「USB 層で何か起きたのか」がカーネルログでしか判定できない
  (`deviceState` は約 2Hz で、SMU が止まるのは 0.5s 未満の出来事なので瞬断は写らない)。
  08-30 は偶然 `dmesg -wT` を流していたから crash 前の USB イベントがゼロだと確認できたが、
  常駐プロセスは再起動で消えるので、**落ちる側で残す**。

  ⚠ 例外は投げない。保存できなくても modeld の終了処理を止めない。
  """
  try:
    out = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10).stdout
    d = CRASH_DIR
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    path = d / f"dmesg-{ts}{tag}.log"
    # ⚠ c4 の dmesg は wlan のビーコン処理ログでほぼ埋まる (08-30 実測: 末尾 200KB の 9 割以上)。
    #   USB / GPU の行が押し出されて使い物にならないので落とす。
    path.write_text("\n".join(ln for ln in out.splitlines() if "wlan:" not in ln)[-200000:] + "\n")
    cloudlog.warning(f"save_dmesg_snapshot: {path}")
    return path
  except Exception:
    cloudlog.exception("save_dmesg_snapshot failed")
    return None


def reset_chestnut(settle: float = 1.0, deep: bool = False) -> bool:
  """chestnut (eGPU) の USB デバイスをリセットする (GS450h 追加)。

  ⚠ tinygrad の AMD-over-USB backend が GPU ハング ("Wait timeout: 3000 ms!") を起こすと、
  modeld を殺し直すだけでは復帰しないことがある。08-29 の実走では
  「車を再起動するまで Big Model Failed のまま」だった。

  ⚠⚠ **08-30 実車で判明: `authorized` 0->1 は使ってはいけない**。chestnut は USB 経由で
  PCIe をトンネルするデバイス (`pcieLtssm` があるのはそのため) で、**再列挙すると PCIe が
  張り直されず `no pcie` で二度とロードできなくなる**。実際 08-30 の実ハングで
  `eGPU model load failed: No interface for AMD:0 is available / RuntimeError('no pcie')` に陥り、
  **車を再起動するまで復帰しなかった**。⚠ 08-29 に offroad で「両方やってもデバイスを失わない」
  ことは確認したが、**あれは GPU が正常なときのテスト**でハング後の PCIe 再確立は見ていなかった。
  ⇒ **`deep` は既定 False**。使うのは「電源断以外に手が無いと分かっている」場合だけ。

  手段は 2 つある:
    1. USBDEVFS_RESET — devnode (/dev/bus/usb/<bus>/<dev>) は root グループで rw なので
       **sudo 不要**。ドライバを外さない軽いリセット。
    2. sysfs の authorized 0 -> 1 (`deep=True`) — ⚠⚠ **PCIe を壊すので既定では使わない**。

  ⚠ ioctl は GPU がハングしていても成功を返すので、戻り値では効いたか判定できない。
  それでも 2 を既定にしないのは、上のとおり **2 は状況を悪化させる**ため。

  ⚠ 例外は投げない。呼び出し元は「落ちる直前 / ロード失敗時の保険」として使い、
  戻り値を見て次の手 (プロセス終了 -> manager による再起動) に進む。
  """
  d = chestnut_device_path()
  if d is None:
    cloudlog.warning("reset_chestnut: chestnut not found in sysfs, skipping")
    return False

  did = False
  node = _chestnut_devnode(d)
  if node is not None and os.access(node, os.W_OK):
    try:
      fd = os.open(node, os.O_WRONLY)
      try:
        fcntl.ioctl(fd, USBDEVFS_RESET, 0)
      finally:
        os.close(fd)
      time.sleep(settle)
      cloudlog.warning(f"reset_chestnut: USBDEVFS_RESET on {node}")
      did = True
    except Exception:
      cloudlog.exception(f"reset_chestnut: ioctl failed on {node}")

  if deep:
    auth = d / "authorized"
    try:
      for val in ("0", "1"):
        subprocess.run(["sudo", "-n", "tee", str(auth)], input=val, text=True,
                       check=True, capture_output=True, timeout=10)
        time.sleep(settle)
      cloudlog.warning(f"reset_chestnut: re-enumerated {d.name} via authorized")
      did = True
    except Exception:
      cloudlog.exception(f"reset_chestnut: authorized reset failed on {d}")

  return did

def chestnut_compiled() -> bool:
  return Path(get_manifest_path(modeld_pkl_path(chestnut=True))).is_file()


# ---- stdout/stderr のリングバッファ (GS450h 追加、08-30) ----

# 直近の print を貯めるリング。capture_stdio() を呼ぶまでは None (= 何もしない)。
_STDIO_RING: collections.deque | None = None

# 保存時に「GPU が何を報告したか」として cloudlog にも上げる行の目印。
# tinygrad ops_amd.py の interrupt_handler() が print する語 + ハング検出の文言。
GPU_FAULT_MARKERS = ("sq_intr", "UTCL2", "IH (", "Wait timeout", "on_device_hang", "MEMVIOL",
                     "ILLEGAL_INST", "EDC_FUE", "EDC_FED")


class _StdioTee(io.TextIOBase):
  """書き込みを元のストリームへ流しつつ、直近の行を手元のリングに残す。

  ⚠⚠ tinygrad の AMD-over-USB backend は、GPU のエラー割り込み (MEMVIOL / ILLEGAL_INST /
  EDC_FUE / EDC_FED / UTCL2_FAULT = ページフォルト) を **print で報告する**
  (`ops_amd.py` の `on_device_hang()` -> `_collect_interrupts()` -> `interrupt_handler()`)。
  ところが modeld の stdout は manager の pty 経由でどこにも残らず、journald も
  永続化されていない。08-30 に swaglog と journal を grep して **`sq_intr` / `UTCL2` /
  `IH (` が 1 件も無い**ことを確認した ⇒ **「なぜハングしたか」を GPU が毎回教えている
  のに、こちらが 1 行も読めていなかった**。cloudlog に流れるのは Python の traceback
  (「Wait timeout: 3000 ms!」まで) だけで、割り込みの種別は入らない。

  ⚠ 常時ファイルに書くのではなくリングに貯めるだけにしてある。落ちたときだけ
  save_stdio_snapshot() で吐く (dmesg スナップショットと同じ流儀)。
  """

  def __init__(self, stream, ring: collections.deque):
    self.stream = stream
    self.ring = ring
    self._partial = ""

  def write(self, s: str) -> int:
    try:
      self.stream.write(s)
    except Exception:
      pass
    try:
      self._partial += s
      while (i := self._partial.find("\n")) >= 0:
        self.ring.append(self._partial[:i])
        self._partial = self._partial[i + 1:]
      if len(self._partial) > 4096:   # 改行が来ないまま膨らませない
        self.ring.append(self._partial)
        self._partial = ""
    except Exception:
      pass
    return len(s)

  def pending(self) -> str:
    return self._partial

  def flush(self) -> None:
    try:
      self.stream.flush()
    except Exception:
      pass

  def writable(self) -> bool:
    return True

  def fileno(self) -> int:
    return self.stream.fileno()

  def isatty(self) -> bool:
    try:
      return self.stream.isatty()
    except Exception:
      return False

  @property
  def buffer(self):
    return getattr(self.stream, "buffer", None)


def capture_stdio(maxlen: int = 2000) -> None:
  """print されるだけで捨てられている出力を、落ちたときに残せるよう手元に貯める。

  ⚠ 二重に包まないよう、既に貯めていれば何もしない。
  ⚠ 例外は投げない。これは modeld の起動時に呼ばれるので、失敗しても起動は続けさせる。
  """
  global _STDIO_RING
  if _STDIO_RING is not None:
    return
  try:
    ring: collections.deque = collections.deque(maxlen=maxlen)
    sys.stdout = _StdioTee(sys.stdout, ring)
    sys.stderr = _StdioTee(sys.stderr, ring)
    _STDIO_RING = ring
  except Exception:
    # 落ちる寸前の保険が、modeld の起動を妨げては本末転倒。
    cloudlog.exception("capture_stdio failed")


def save_stdio_snapshot(tag: str = "") -> Path | None:
  """リングに貯めた stdout/stderr を crash ログの隣に残す (GS450h 追加、08-30)。

  ⚠ GPU の割り込みらしい行は cloudlog にも上げる。crash ファイルは回収し忘れるが、
  swaglog は走行のたびに引いているので、そちらだけで種別が分かるようにしておく。

  ⚠ 例外は投げない。保存できなくても modeld の終了処理を止めない。
  """
  if _STDIO_RING is None:
    return None
  try:
    lines = list(_STDIO_RING)
    for s in (sys.stdout, sys.stderr):
      if isinstance(s, _StdioTee) and s.pending():
        lines.append(s.pending())
    if not lines:
      return None

    hits = [ln for ln in lines if any(m in ln for m in GPU_FAULT_MARKERS)]
    if hits:
      cloudlog.error(f"gpu faults before exit{tag}: {' | '.join(hits[-12:])[:2000]}")

    d = CRASH_DIR
    d.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
    path = d / f"stdio-{ts}{tag}.log"
    path.write_text("\n".join(lines) + "\n")
    cloudlog.warning(f"save_stdio_snapshot: {path} ({len(lines)} lines, {len(hits)} gpu faults)")
    return path
  except Exception:
    cloudlog.exception("save_stdio_snapshot failed")
    return None


def chestnut_ready(state) -> bool:
  return state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE and not state.supplyFault and state.pcieLtssm == CHESTNUT_PCIE_READY
