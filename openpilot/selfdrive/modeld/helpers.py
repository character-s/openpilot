import fcntl
import io
import json
import os
import pickle
import shutil
import struct
import subprocess
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


def reset_chestnut(settle: float = 1.0, deep: bool = True) -> bool:
  """chestnut (eGPU) の USB デバイスをリセットする (GS450h 追加)。

  ⚠ tinygrad の AMD-over-USB backend が GPU ハング ("Wait timeout: 3000 ms!") を起こすと、
  modeld を殺し直すだけでは復帰しないことがある。08-29 の実走では
  「車を再起動するまで Big Model Failed のまま」だった。

  手段は 2 つあり、**既定では両方続けて実行する**:
    1. USBDEVFS_RESET — devnode (/dev/bus/usb/<bus>/<dev>) は root グループで rw なので
       **sudo 不要**。ドライバを外さない軽いリセット。
    2. sysfs の authorized 0 -> 1 — 切り離して再認証する強い手。root 専用なので sudo 経由
       (openpilot は User=comma で動く)。

  ⚠⚠ 1 だけにしないのは、**GPU がハングしていても ioctl は成功を返す**ため。戻り値では
  効いたか判定できず、1 を繰り返すだけで復帰しない恐れがある。08-29 に実機 (offroad) で
  2 の安全性も確認済み: deauthorize 中も sysfs エントリは残り、再認証で devnum ごと復帰する。

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


def chestnut_ready(state) -> bool:
  return state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE and not state.supplyFault and state.pcieLtssm == CHESTNUT_PCIE_READY
