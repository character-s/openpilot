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
from openpilot.common.hardware.usb import read_int
from openpilot.common.hardware.hw import Paths
from openpilot.common.swaglog import cloudlog

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
CHESTNUT_POWERED_VOLTAGE = 5000
CHESTNUT_PCIE_READY = 0x78

# 落ちたときのスナップショット置き場 (sentry の crash ログと同じ)。テストから差し替える。
CRASH_DIR = Path(Paths.crash_log_root())  # c4 では /data/community/crashes (PC では ~/.comma/community/crashes)


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


# _IO('U', 20)。openpilot/system/hardware/chestnut/flash.py と同じ値 (import すると CI の _toplevel_names に
# 載らないので代入で持つ)。devnode に対して効くので comma ユーザーのままで ioctl できる (sysfs の authorized と違う)。
USBDEVFS_RESET = 0x5514


def _chestnut_devnode(d: Path) -> Path | None:
  bus, dev = read_int(d / "busnum"), read_int(d / "devnum")
  return Path(f"/dev/bus/usb/{bus:03d}/{dev:03d}") if bus and dev else None


def _write_crash_file(kind: str, tag: str, text: str, note: str = "") -> Path:
  CRASH_DIR.mkdir(parents=True, exist_ok=True)
  ts = datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
  path = CRASH_DIR / f"{kind}-{ts}{tag}.log"
  path.write_text(text + "\n")
  cloudlog.warning(f"save_{kind}_snapshot: {path}{note}")
  return path


def save_dmesg_snapshot(tag: str = "") -> Path | None:
  """クラッシュ直前のカーネルログを crash ログの隣に残す (GS450h 追加)。

  ⚠ journald は永続化されていないので再起動で dmesg は消える。chestnut のハングが USB 層の出来事かは
     カーネルログでしか判定できない (deviceState は約 2Hz で瞬断は写らない) ので、落ちる側で残す。
  ⚠ 例外は投げない。保存できなくても modeld の終了処理を止めない。
  """
  try:
    out = subprocess.run(["dmesg", "-T"], capture_output=True, text=True, timeout=10).stdout
    # c4 の dmesg は wlan のビーコン処理ログでほぼ埋まり USB / GPU の行が押し出されるので落とす。
    return _write_crash_file("dmesg", tag, "\n".join(ln for ln in out.splitlines() if "wlan:" not in ln)[-200000:])
  except Exception:
    cloudlog.exception("save_dmesg_snapshot failed")
    return None


def reset_chestnut(settle: float = 1.0) -> bool:
  """chestnut (eGPU) を USBDEVFS_RESET で軽くリセットする (GS450h 追加)。

  GPU ハング ("Wait timeout") 後は modeld を殺し直すだけでは復帰しないための保険。
  ⚠⚠ sysfs `authorized` 0->1 の再列挙は使わない: USB 越しの PCIe が張り直されず `no pcie` で
     二度とロードできなくなる。offroad の正常時テストで通っても、ハング後の PCIe 再確立は別。
  ⚠ ioctl はハング中でも成功を返すので戻り値は「叩けたか」であって「効いたか」ではない。
  ⚠ 例外は投げない。呼び出し元は次の手 (プロセス終了 -> manager 再起動) に進む。
  """
  d = chestnut_device_path()
  if d is None:
    cloudlog.warning("reset_chestnut: chestnut not found in sysfs, skipping")
    return False
  node = _chestnut_devnode(d)
  if node is None or not os.access(node, os.W_OK):
    return False
  try:
    fd = os.open(node, os.O_WRONLY)
    try:
      fcntl.ioctl(fd, USBDEVFS_RESET, 0)
    finally:
      os.close(fd)
    time.sleep(settle)
    cloudlog.warning(f"reset_chestnut: USBDEVFS_RESET on {node}")
    return True
  except Exception:
    cloudlog.exception(f"reset_chestnut: ioctl failed on {node}")
    return False

def chestnut_compiled() -> bool:
  return Path(get_manifest_path(modeld_pkl_path(chestnut=True))).is_file()


# ---- stdout/stderr のリングバッファ (GS450h 追加) ----

# 直近の print を貯めるリング。capture_stdio() を呼ぶまでは None (= 何もしない)。
_STDIO_RING: collections.deque | None = None

# 保存時に「GPU が何を報告したか」として cloudlog にも上げる行の目印。
# tinygrad ops_amd.py の interrupt_handler() が print する語 + ハング検出の文言。
GPU_FAULT_MARKERS = ("sq_intr", "UTCL2", "IH (", "Wait timeout", "on_device_hang", "MEMVIOL",
                     "ILLEGAL_INST", "EDC_FUE", "EDC_FED")


class _StdioTee(io.TextIOBase):
  """書き込みを元のストリームへ流しつつ、直近の行を手元のリングに残す。

  ⚠ tinygrad の AMD-over-USB backend は GPU のエラー割り込み (MEMVIOL / UTCL2_FAULT 等) を print で
     報告する (`ops_amd.py` の `interrupt_handler()`) が、modeld の stdout は manager の pty 経由で
     どこにも残らず cloudlog には traceback しか流れない。種別を残すにはここで拾うしかない。
  ⚠ 常時ファイルに書くのではなくリングに貯めるだけ。落ちたときだけ save_stdio_snapshot() で吐く。
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
      *lines, self._partial = (self._partial + s).split("\n")
      self.ring.extend(lines)
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
  """リングに貯めた stdout/stderr を crash ログの隣に残す (GS450h 追加)。

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

    return _write_crash_file("stdio", tag, "\n".join(lines), f" ({len(lines)} lines, {len(hits)} gpu faults)")
  except Exception:
    cloudlog.exception("save_stdio_snapshot failed")
    return None


def chestnut_ready(state) -> bool:
  return state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE and not state.supplyFault and state.pcieLtssm == CHESTNUT_PCIE_READY
