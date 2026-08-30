#!/usr/bin/env python3
"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from collections.abc import Callable
import os
import traceback
os.environ['GMMU'] = '0'
import numpy as np
import threading
import time
from setproctitle import setproctitle
from tinygrad.tensor import Tensor

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware import COMMA_HARDWARE
from openpilot.selfdrive.modeld.helpers import chestnut_present, load_oob, reset_chestnut, save_dmesg_snapshot, capture_stdio, save_stdio_snapshot
from openpilot.cereal import log
from opendbc.car.structs import car
from openpilot.cereal.services import SERVICE_LIST
from openpilot.cereal.messaging import PubMaster, SubMaster
from openpilot.cereal.visionipc import VisionStreamType
from msgq.visionipc import VisionIpcClient, VisionBuf
from opendbc.car.car_helpers import get_demo_car_params
from openpilot.common.file_chunker import open_file_chunked
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import config_realtime_process, DT_MDL
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.common.transformations.model import get_warp_matrix
from openpilot.system import sentry
from openpilot.system.camerad.cameras.nv12_info import get_nv12_info
from openpilot.selfdrive.controls.lib.desire_helper import DesireHelper
from openpilot.selfdrive.controls.lib.drive_helpers import get_accel_from_plan, smooth_value
from openpilot.selfdrive.modeld.modeld import ChestnutState

from openpilot.sunnypilot.modeld_v2.fill_model_msg import fill_model_msg, fill_pose_msg, PublishState, get_curvature_from_output
from openpilot.sunnypilot.modeld_v2.constants import Plan
from openpilot.sunnypilot.modeld_v2.meta_helper import load_meta_constants
from openpilot.sunnypilot.modeld_v2.camera_offset_helper import CameraOffsetHelper
from openpilot.sunnypilot.modeld_v2.compile_modeld import derive_frame_skip, make_split_input_queues, make_supercombo_input_queues, WARP_INPUTS, POLICY_INPUTS
from openpilot.sunnypilot.livedelay.helpers import get_lat_delay
from openpilot.sunnypilot.modeld_v2.modeld_base import ModelStateBase
from openpilot.sunnypilot.models.helpers import get_active_bundle
from openpilot.sunnypilot.selfdrive.controls.lib.relc import RoadEdgeLaneChangeController

PROCESS_NAME = "openpilot.selfdrive.modeld.modeld_tinygrad"
BIG_MODEL_TIMEOUT = 150  # GS450h: measured 80.3s cold load on chestnut (2026-08-24); 60s always timed out

# Cap on the bundle's "long" override, which is a first-order lag applied to the model's
# desiredAcceleration *after* it is produced (see smooth_value in drive_helpers.py).
# TT ships long=".3". With DT_MDL=0.05 that is alpha = 1-exp(-0.05/0.3) = 0.154, so a step to
# 1.0 m/s^2 only reaches 0.15 in the first frame and 0.63 after 0.3s. GS 450h's
# longitudinalActuatorDelay is 0.05s, so this filter - not the actuator and not the model -
# dominates the onset. That is the "soft pedal every time" the driver reported (2026-08-25).
# ⚠ This is a cap, not a fixed value: a bundle asking for less keeps its own number.
# ⚠ modeld also feeds LONG_SMOOTH_SECONDS into long_delay, so the planner's delay
#   compensation follows this automatically - do not compensate for it a second time.
# 0.15 per the driver's call (2026-08-25, ".15 or .2 is fine"): step response reaches 49%
# in 0.1s / 74% in 0.2s, vs 28%/49% at the shipped 0.3 - most of the win with less risk
# of surging on plan noise than 0.1 would carry.
LONG_SMOOTH_SECONDS_MAX = 0.15

# tinygrad's AMD-over-USB backend guards the device with a flock on /tmp/am_usb:<bus>-<port>.lock
# and takes it with LOCK_EX | LOCK_NB (system.py:144-148) - whoever asks second fails instantly
# with "Failed to acquire lock file", which surfaces as "No interface for AMD:0 is available".
# Measured 2026-08-25: two boots that decided at since_boot 34.0/34.6s loaded fine, the one that
# decided at 35.3s lost the lock. Losing that race cost an entire drive's worth of driving model.
# Nothing else in openpilot takes this lock (chestnut_present() and usb.py both read sysfs only),
# so the contender is another modeld instance that has not exited yet - retrying is the fix.
EGPU_LOAD_ATTEMPTS = 5  # total attempts, not retries
EGPU_LOCK_RETRY_WAIT = 3.0  # [s] between attempts, well inside BIG_MODEL_TIMEOUT


def _is_lock_contention(e: BaseException) -> bool:
  # the real error is nested inside an ExceptionGroup, so match on the rendered text
  return "Failed to acquire lock file" in repr(e)


def _egpu_lock_holder() -> str:
  """Who is holding the am_usb lock. Diagnosis only - never raises.

  ⚠⚠ **`fuser` は「ファイルを開いているプロセス」しか教えてくれない**。flock を実際に
  保持しているかは `/proc/locks` にしか出ない。08-30 に fuser の出力だけを見て
  「自分がロックを握っている」と読んだが、**開いているだけの可能性と区別できていなかった**
  (tinygrad の `flock_acquire` は `os.open` してから flock するので、**flock に失敗しても
  fd は開いたまま残る**。しかも fd を `System` シングルトンの属性に上書き代入するため、
  2 回目以降は前の fd が閉じられずに漏れる)。⇒ 3 つを分けて出す:

    my_fds        自分が開いている lock ファイルの fd (漏れの数)
    flock_held_by flock を **実際に保持している** PID (me / other)
    fuser         開いているだけのプロセス一覧 (従来の情報)
  """
  import glob
  import subprocess
  try:
    locks = glob.glob("/tmp/am_usb:*.lock")
    if not locks:
      return "no lock file"
    me = os.getpid()
    parts = [f"pid={me}"]

    mine = []
    for fd in os.listdir(f"/proc/{me}/fd"):
      try:
        if "am_usb" in os.readlink(f"/proc/{me}/fd/{fd}"):
          mine.append(fd)
      except OSError:
        pass
    parts.append(f"my_fds={mine}")

    inodes = set()
    for path in locks:
      try:
        inodes.add(os.stat(path).st_ino)
      except OSError:
        pass
    held = []
    with open("/proc/locks") as f:
      for line in f:
        # 例: "1: FLOCK  ADVISORY  WRITE 1234 08:01:12345 0 EOF"  (待機中の行は "-> " が入る)
        cols = line.replace("->", " ").split()
        if len(cols) < 6 or cols[1] != "FLOCK":
          continue
        ino = cols[5].rsplit(":", 1)[-1]
        if ino.isdigit() and int(ino) in inodes:
          held.append(f"{cols[4]}({'me' if cols[4] == str(me) else 'other'})")
    parts.append(f"flock_held_by={held or 'nobody'}")

    out = subprocess.run(["fuser", "-v"] + locks, capture_output=True, text=True, timeout=5)
    parts.append("fuser=" + (" ".join((out.stdout + out.stderr).split())[:150] or "nobody"))
    return " ".join(parts)
  except Exception:
    return "unknown"


def _load_with_retry(make_model, attempts: int = EGPU_LOAD_ATTEMPTS, wait: float = EGPU_LOCK_RETRY_WAIT):
  """Call make_model up to `attempts` times, retrying only on am_usb lock contention.

  Returns (model, error); model is None when every attempt failed.

  ⚠⚠ 返すのは **1 回目の例外**。tinygrad は候補インターフェースを順に試し、1 つ目で
  am_usb ロックを取ったまま初期化に失敗すると、2 つ目以降が **自分が残したロック**と
  ぶつかる。最後の例外を返すと **本当の失敗理由が毎回「ロック競合」に化ける**
  (08-30 に踏んだ: eGPU が 10 連敗したログが全部 `lock holder = 自分自身` で、
  「なぜ GPU を掴めなかったのか」が 1 件も残っていなかった)。
  """
  first: Exception | None = None
  cloudlog.warning(f"eGPU load starting (lock holder before: {_egpu_lock_holder()})")
  for attempt in range(1, attempts + 1):
    try:
      return make_model(), None
    except Exception as e:  # an unhandled exception in the load thread would die silently
      if first is None:
        first = e
        # ⚠ ExceptionGroup は repr だと 1 行に潰れ、どの経路がどこで落ちたかはサブ例外の
        #   traceback にしか無い。1 回目だけ全部展開して残す。
        cloudlog.error("eGPU load attempt 1 failed:\n" + "".join(traceback.format_exception(e)))
      if not _is_lock_contention(e) or attempt == attempts:
        break
      cloudlog.warning(f"eGPU lock held (attempt {attempt}/{attempts}), retry in {wait}s (holder: {_egpu_lock_holder()})")
      time.sleep(wait)
  cloudlog.error(f"eGPU model load failed (lock holder: {_egpu_lock_holder()})")
  return None, first


def _pkl_exists(path):
  from openpilot.common.file_chunker import get_manifest_path
  return os.path.exists(path) or os.path.exists(get_manifest_path(path))


def _find_driving_pkl(bundle):
  if (override := os.environ.get('COMBINED_MODEL_PKL')) and _pkl_exists(override):
    return override
  if bundle is None or not bundle.models:
    return None
  from openpilot.common.hardware.hw import Paths
  model_root = Paths.model_root()

  pkl_name = bundle.models[0].artifact.fileName
  pkl_path = os.path.join(model_root, pkl_name)
  if _pkl_exists(pkl_path):
    return pkl_path
  return None


class FrameMeta:
  frame_id: int = 0
  timestamp_sof: int = 0
  timestamp_eof: int = 0

  def __init__(self, vipc=None):
    if vipc is not None:
      self.frame_id, self.timestamp_sof, self.timestamp_eof = vipc.frame_id, vipc.timestamp_sof, vipc.timestamp_eof


class ModelState(ModelStateBase):
  inputs: dict[str, np.ndarray]
  prev_desire: np.ndarray

  def __init__(self, cam_w: int, cam_h: int, chestnut: bool = False):
    ModelStateBase.__init__(self)

    env_pkl = os.environ.get('COMBINED_MODEL_PKL')
    if env_pkl and os.path.exists(env_pkl):
      model_bundle = None
    else:
      model_bundle = get_active_bundle(chestnut=chestnut)
    self.generation = model_bundle.generation if model_bundle is not None else None
    overrides = {override.key: override.value for override in model_bundle.overrides} if model_bundle else {}

    self.LAT_SMOOTH_SECONDS = float(overrides.get('lat', ".0"))
    self.LONG_SMOOTH_SECONDS = min(float(overrides.get('long', ".0")), LONG_SMOOTH_SECONDS_MAX)
    self.MIN_LAT_CONTROL_SPEED = 0.3
    self.PLANPLUS_CONTROL: float = 1.0
    self.chestnut = chestnut

    pkl_path = _find_driving_pkl(model_bundle)
    assert pkl_path is not None, f"No driving pkl found for {'chestnut' if chestnut else 'small model'} — all models must be compiled with compile_modeld.py"
    self._init_combined(pkl_path, cam_w, cam_h, model_bundle)

  def _init_combined(self, pkl_path, cam_w, cam_h, bundle):
    cloudlog.warning(f"loading combined pkl: {pkl_path}")
    jits = load_oob(open_file_chunked(pkl_path))

    metadata = jits['metadata']
    self.WARP_DEV = metadata.get('warp_dev', 'QCOM' if COMMA_HARDWARE else 'CPU')
    self.DEV = 'AMD' if self.chestnut else ('QCOM' if COMMA_HARDWARE else 'CPU')
    self.QUEUE_DEV = self.DEV
    self.run_policy = jits['run_policy']
    self.warp = jits[(cam_w, cam_h)]

    if 'model' in metadata:
      model_metadata = metadata['model']
      self.vision_output_slices = model_metadata['output_slices']
      self.policy_output_slices = {}
      self._policy_slices_list = []
      self._combined_model_type = 'supercombo'
      self._vision_input_names = [key for key in model_metadata['input_shapes'] if 'img' in key]
      frame_skip = derive_frame_skip({}, model_metadata['input_shapes'])
      self.input_queues, self.numpy_inputs = make_supercombo_input_queues(model_metadata['input_shapes'],
                                                                          frame_skip, device=self.QUEUE_DEV)
    else:
      vision_metadata = metadata['vision']
      policy_keys = [k for k in metadata if k != 'vision']
      if policy_keys == ['policy']:
        self._combined_model_type = 'split'
      else:
        self._combined_model_type = 'multi_policy'
      self.vision_output_slices = vision_metadata['output_slices']
      self._policy_keys = policy_keys
      self._policy_slices_list = [metadata[k]['output_slices'] for k in policy_keys]
      self.policy_output_slices = self._policy_slices_list[0]
      self._has_on_policy = any('on' in k.lower() for k in policy_keys)
      self._vision_input_names = [key for key in vision_metadata['input_shapes'] if 'img' in key]
      first_policy_meta = metadata[policy_keys[0]]
      frame_skip = derive_frame_skip(vision_metadata['input_shapes'], first_policy_meta['input_shapes'])
      self.input_queues, self.numpy_inputs = make_split_input_queues(vision_metadata['input_shapes'],
                                                                     first_policy_meta['input_shapes'],
                                                                     frame_skip, device=self.QUEUE_DEV)

    self._desire_key = next(key for key in self.numpy_inputs if key.startswith('desire'))
    self._road_key = next(key for key in self._vision_input_names if 'big' not in key)
    self._wide_key = next(key for key in self._vision_input_names if 'big' in key)

    is_20hz = bundle.is20hz if bundle else self._combined_model_type in ('split', 'multi_policy')
    if is_20hz:
      from openpilot.sunnypilot.models.split_model_constants import SplitModelConstants
      self.constants = SplitModelConstants()
    else:
      from openpilot.sunnypilot.modeld_v2.constants import ModelConstants
      self.constants = ModelConstants()

    if self._combined_model_type != 'supercombo':
      from openpilot.sunnypilot.modeld_v2.parse_model_outputs_split import Parser as SplitParser
      self.parser = SplitParser()
    else:
      from openpilot.sunnypilot.modeld_v2.parse_model_outputs import Parser as CombinedParser
      self.parser = CombinedParser()

    self.prev_desire = np.zeros(self.constants.DESIRE_LEN, dtype=np.float32)
    self.full_frames: dict = {}
    self._blob_cache: dict = {}
    nv12_info = get_nv12_info(cam_w, cam_h)
    self.frame_buf_params = dict.fromkeys(self._vision_input_names, nv12_info)

    yuv_size = self.frame_buf_params[self._road_key][3]
    frame_tensor = Tensor(np.zeros(yuv_size, dtype=np.uint8), device=self.WARP_DEV).contiguous().realize()
    big_frame_tensor = Tensor(np.zeros(yuv_size, dtype=np.uint8), device=self.WARP_DEV).contiguous().realize()
    self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=frame_tensor, big_frame=big_frame_tensor)

  def warmup(self) -> None:
    dummy_frames = {k: np.zeros(self.frame_buf_params[k][3], dtype=np.uint8) for k in self._vision_input_names}
    transforms = {k: np.eye(3, dtype=np.float32) for k in [self._road_key, self._wide_key] if k}

    dummy_inputs = {}
    for k, v in self.numpy_inputs.items():
      if k not in ['tfm', 'big_tfm', 'prev_feat']:
        dummy_inputs[k] = np.zeros(v.shape, dtype=v.dtype)

    self.run(dummy_frames, transforms, dummy_inputs, prepare_only=False)

    for v in self.numpy_inputs.values():
      v[:] = 0
    self.prev_desire[:] = 0
    self.full_frames.clear()
    self._blob_cache.clear()


  @property
  def mlsim(self) -> bool:
    return bool(self.generation is not None and self.generation >= 11)

  @property
  def vision_input_names(self) -> list[str]:
    return self._vision_input_names

  @property
  def desire_key(self) -> str:
    return self._desire_key

  def run(self, bufs: dict[str, VisionBuf], transforms: dict[str, np.ndarray],
          inputs: dict[str, np.ndarray], prepare_only: bool,
          after_enqueue: Callable[[], None] | None = None) -> dict[str, np.ndarray] | None:
    for key in bufs.keys():
      ptr = np.frombuffer(bufs[key].data, dtype=np.uint8).ctypes.data
      yuv_size = self.frame_buf_params[key][3]
      cache_key = (key, ptr)
      if cache_key not in self._blob_cache:
        self._blob_cache[cache_key] = Tensor.from_blob(ptr, (yuv_size,), dtype='uint8', device=self.WARP_DEV)
      self.full_frames[key] = self._blob_cache[cache_key]

    desire_key = self.desire_key
    inputs[desire_key][0] = 0
    self.numpy_inputs[desire_key][:] = np.where(inputs[desire_key] - self.prev_desire > .99, inputs[desire_key], 0)
    self.prev_desire[:] = inputs[desire_key]
    for key in ('traffic_convention', 'lateral_control_params', 'action_t'):
      if key in self.numpy_inputs and key in inputs:
        self.numpy_inputs[key][:] = inputs[key]

    road_key = self._road_key
    wide_key = self._wide_key
    self.numpy_inputs['tfm'][:, :] = transforms[road_key].reshape(3, 3)
    self.numpy_inputs['big_tfm'][:, :] = transforms[wide_key].reshape(3, 3)

    if prepare_only:
      self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[road_key], big_frame=self.full_frames[wide_key])
      return None
    warped = self.warp(**{k: self.input_queues[k] for k in WARP_INPUTS}, frame=self.full_frames[road_key], big_frame=self.full_frames[wide_key])
    raw_outputs = self.run_policy(**{k: self.input_queues[k] for k in POLICY_INPUTS if k in self.input_queues}, warped=warped)
    if after_enqueue is not None:
      after_enqueue()

    if self._combined_model_type == 'supercombo':
      model_output = raw_outputs.numpy().flatten()
      if self.chestnut and not np.all(np.isfinite(model_output)):
        raise RuntimeError("model output not finite")
      sliced = {k: model_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_outputs(sliced)
      if 'prev_feat' in self.numpy_inputs:
        self.numpy_inputs['prev_feat'][:] = model_output[self.vision_output_slices['hidden_state']]
    else:
      vision_output = raw_outputs[0].numpy().flatten()
      vision_sliced = {k: vision_output[np.newaxis, v] for k, v in self.vision_output_slices.items()}
      outputs = self.parser.parse_vision_outputs(vision_sliced)

      if 'prev_feat' in self.numpy_inputs and 'hidden_state' in self.vision_output_slices:
        self.numpy_inputs['prev_feat'][:] = vision_output[self.vision_output_slices['hidden_state']]

      for i, policy_slices in enumerate(self._policy_slices_list):
        policy_output = raw_outputs[i + 1].numpy().flatten()
        policy_sliced = {k: policy_output[np.newaxis, v] for k, v in policy_slices.items()}
        parsed = self.parser.parse_policy_outputs(policy_sliced)
        if ('off' in self._policy_keys[i]
          and self._has_on_policy
          and any('plan' in self._policy_slices_list[j] for j, k in enumerate(self._policy_keys) if 'on' in k.lower())):

          parsed.pop('plan', None)

        outputs.update(parsed)

      if 'planplus' in outputs and 'plan' in outputs:
        outputs['plan'] = outputs['plan'] + outputs['planplus']

    if 'desired_curvature' in outputs and 'prev_desired_curv' in self.numpy_inputs:
      buf = self.numpy_inputs['prev_desired_curv']
      buf[0, :-1] = buf[0, 1:]
      buf[0, -1, :] = outputs['desired_curvature'][0, :] if not self.mlsim else 0

    return outputs

  def get_action_from_model(self, model_output: dict[str, np.ndarray], prev_action: log.ModelDataV2.Action,
                            lat_action_t: float, long_action_t: float, v_ego: float) -> log.ModelDataV2.Action:
    if 'action' not in model_output:
      plan = model_output['plan'][0]
      desired_accel = get_accel_from_plan(plan[:, Plan.VELOCITY][:, 0], plan[:, Plan.ACCELERATION][:, 0], self.constants.T_IDXS,
                                          action_t=long_action_t)

      curvature_plan = (plan + (self.PLANPLUS_CONTROL - 1.0) * model_output['planplus'][0]
                        if 'planplus' in model_output and self.PLANPLUS_CONTROL != 1.0 else plan)
      desired_curvature = get_curvature_from_output(model_output, curvature_plan, v_ego, lat_action_t, self.mlsim)
    else:
      desired_accel = model_output['action'][0, 1]
      desired_curvature = model_output['action'][0, 0] / (max(1.0, v_ego))**2

    stop = v_ego < 0.3 and desired_accel < 0.1
    desired_accel = smooth_value(desired_accel, prev_action.desiredAcceleration, self.LONG_SMOOTH_SECONDS)

    if self.generation is not None and self.generation >= 10: # smooth curvature for post FOF models
      if v_ego > self.MIN_LAT_CONTROL_SPEED:
        desired_curvature = smooth_value(desired_curvature, prev_action.desiredCurvature, self.LAT_SMOOTH_SECONDS)
      else:
        desired_curvature = prev_action.desiredCurvature

    return log.ModelDataV2.Action(desiredCurvature=float(desired_curvature), desiredAcceleration=float(desired_accel), shouldStop=bool(stop))


def main(demo=False):
  cloudlog.warning("modeld init")

  sentry.set_tag("daemon", PROCESS_NAME)
  cloudlog.bind(daemon=PROCESS_NAME)
  setproctitle(PROCESS_NAME)
  config_realtime_process(7, 54)

  CHESTNUT = chestnut_present()
  cloudlog.event("modeld eGPU decision", chestnut=CHESTNUT,
                 bundle=getattr(get_active_bundle(chestnut=CHESTNUT), 'internalName', None),
                 since_boot=round(time.monotonic(), 1))
  if CHESTNUT:
    os.environ['HCQDEV_WAIT_TIMEOUT_MS'] = '3000'

  params = Params()
  params.put_bool("ChestnutLoading", CHESTNUT)
  params.remove("ChestnutActive")

  # visionipc clients
  while True:
    available_streams = VisionIpcClient.available_streams("camerad", block=False)
    if available_streams:
      use_extra_client = VisionStreamType.VISION_STREAM_WIDE_ROAD in available_streams and VisionStreamType.VISION_STREAM_NARROW_ROAD in available_streams
      main_wide_camera = VisionStreamType.VISION_STREAM_NARROW_ROAD not in available_streams
      break
    time.sleep(.1)

  vipc_client_main_stream = VisionStreamType.VISION_STREAM_WIDE_ROAD if main_wide_camera else VisionStreamType.VISION_STREAM_NARROW_ROAD
  vipc_client_main = VisionIpcClient("camerad", vipc_client_main_stream, True)
  vipc_client_extra = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_WIDE_ROAD, False)
  cloudlog.warning(f"vision stream set up, main_wide_camera: {main_wide_camera}, use_extra_client: {use_extra_client}")

  while not vipc_client_main.connect(False):
    time.sleep(0.1)
  while use_extra_client and not vipc_client_extra.connect(False):
    time.sleep(0.1)

  cloudlog.warning(f"connected main cam with buffer size: {vipc_client_main.buffer_len} ({vipc_client_main.width} x {vipc_client_main.height})")
  if use_extra_client:
    cloudlog.warning(f"connected extra cam with buffer size: {vipc_client_extra.buffer_len} ({vipc_client_extra.width} x {vipc_client_extra.height})")

  cloudlog.warning("loading model")
  st = time.monotonic()

  model = None
  if CHESTNUT:
    result: list = []

    def make_big():
      m = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=True)
      m.warmup()
      return m

    def load():
      result.append(_load_with_retry(make_big))

    loader = threading.Thread(target=load, daemon=True)
    loader.start()
    loader.join(BIG_MODEL_TIMEOUT + (EGPU_LOAD_ATTEMPTS - 1) * EGPU_LOCK_RETRY_WAIT)
    # read the result exactly once: the thread may still be running after a timeout, and a model
    # that finishes loading late must not swap in mid-drive
    model, load_err = result[0] if result else (None, None)
    params.put_bool("ChestnutActive", model is not None)
    if model is None:
      why = repr(load_err) if load_err else f"no result after {BIG_MODEL_TIMEOUT}s"
      cloudlog.error(f"eGPU model load failed or timed out ({why}); falling back to small model")
      # GS450h: 落ちる直前のカーネルログを残す (再起動で dmesg が消えるため)。
      save_stdio_snapshot("-loadfail")
      save_dmesg_snapshot("-loadfail")
      # GS450h: GPU がハングしたままだと放置すると次のロードも失敗する。small で走り続ける裏で
      # USB を再列挙しておき、次の起動でクリーンなデバイスを掴めるようにする
      # (08-29 まではこれが無く「車を再起動するまで Big Model Failed」だった)。
      reset_chestnut()

  small_model = ModelState(cam_w=vipc_client_main.width, cam_h=vipc_client_main.height, chestnut=False) if model is None or CHESTNUT else None
  if model is None:
    model = small_model
  params.put_bool("ChestnutLoading", False)
  assert model is not None
  cloudlog.warning(f"models loaded in {time.monotonic() - st:.1f}s, modeld starting")

  # messaging
  pub_socks = ["modelV2", "drivingModelData", "cameraOdometry", "modelDataV2SP"] + (["chestnutState"] if CHESTNUT else [])
  pm = PubMaster(pub_socks)
  sm = SubMaster(["deviceState", "carState", "narrowRoadCameraState", "extrinsicsCalibration", "driverMonitoringState", "carControl", "lateralDelay"])

  publish_state = PublishState()
  chestnut_state = ChestnutState(pm, model.chestnut) if CHESTNUT else None

  # setup filter to track dropped frames
  frame_dropped_filter = FirstOrderFilter(0., 10., 1. / model.constants.MODEL_FREQ)
  frame_id = 0
  last_vipc_frame_id = 0
  run_count = 0

  model_transform_main = np.zeros((3, 3), dtype=np.float32)
  model_transform_extra = np.zeros((3, 3), dtype=np.float32)
  live_calib_seen = False
  buf_main, buf_extra = None, None
  meta_main = FrameMeta()
  meta_extra = FrameMeta()
  camera_offset_helper = CameraOffsetHelper()


  if demo:
    CP = get_demo_car_params()
  else:
    CP = messaging.log_from_bytes(params.get("CarParams", block=True), car.CarParams)
  cloudlog.info("modeld got CarParams: %s", CP.brand)

  # TODO Move smooth seconds to action function
  long_delay = CP.longitudinalActuatorDelay + model.LONG_SMOOTH_SECONDS
  prev_action = log.ModelDataV2.Action()

  DH = DesireHelper()
  meta_constants = load_meta_constants()
  RELC = RoadEdgeLaneChangeController()

  while True:
    # Keep receiving frames until we are at least 1 frame ahead of previous extra frame
    while meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
      buf_main = vipc_client_main.recv()
      meta_main = FrameMeta(vipc_client_main)
      if buf_main is None:
        break

    if buf_main is None:
      cloudlog.debug("vipc_client_main no frame")
      continue

    if use_extra_client:
      # Keep receiving extra frames until frame id matches main camera
      while True:
        buf_extra = vipc_client_extra.recv()
        meta_extra = FrameMeta(vipc_client_extra)
        if buf_extra is None or meta_main.timestamp_sof < meta_extra.timestamp_sof + 25000000:
          break

      if buf_extra is None:
        cloudlog.debug("vipc_client_extra no frame")
        continue

      if abs(meta_main.timestamp_sof - meta_extra.timestamp_sof) > 10000000:
        cloudlog.error(f"frames out of sync! main: {meta_main.frame_id} ({meta_main.timestamp_sof / 1e9:.5f}),\
                       extra: {meta_extra.frame_id} ({meta_extra.timestamp_sof / 1e9:.5f})")

    else:
      # Use single camera
      buf_extra = buf_main
      meta_extra = meta_main

    sm.update(0)
    desire = DH.desire
    is_rhd = sm["driverMonitoringState"].isRHD
    frame_id = sm["narrowRoadCameraState"].frameId
    v_ego = max(sm["carState"].vEgo, 0.)
    if sm.frame % 60 == 0:
      model.lat_delay = get_lat_delay(params, sm["lateralDelay"].lateralDelay)
      model.PLANPLUS_CONTROL = params.get("PlanplusControl", return_default=True)
      camera_offset_helper.set_offset(params.get("CameraOffset", return_default=True))
    lat_delay = model.lat_delay + model.LAT_SMOOTH_SECONDS
    if sm.updated["extrinsicsCalibration"] and sm.seen['narrowRoadCameraState'] and sm.seen['deviceState']:
      device_from_calib_euler = np.array(sm["extrinsicsCalibration"].rpyCalib, dtype=np.float32)
      dc = DEVICE_CAMERAS[(str(sm['deviceState'].deviceType), str(sm['narrowRoadCameraState'].sensor))]
      main_intrinsics = dc.wide_road.intrinsics if main_wide_camera else dc.narrow_road.intrinsics
      model_transform_main = get_warp_matrix(device_from_calib_euler, main_intrinsics, False).astype(np.float32)
      model_transform_extra = get_warp_matrix(device_from_calib_euler, dc.wide_road.intrinsics, True).astype(np.float32)
      model_transform_main, model_transform_extra = camera_offset_helper.update(model_transform_main, model_transform_extra, sm, main_wide_camera)
      live_calib_seen = True

    traffic_convention = np.zeros(2)
    traffic_convention[int(is_rhd)] = 1

    vec_desire = np.zeros(model.constants.DESIRE_LEN, dtype=np.float32)
    if desire >= 0 and desire < model.constants.DESIRE_LEN:
      vec_desire[desire] = 1

    # tracked dropped frames
    vipc_dropped_frames = max(0, meta_main.frame_id - last_vipc_frame_id - 1)
    frames_dropped = frame_dropped_filter.update(min(vipc_dropped_frames, 10))
    if run_count < 10: # let frame drops warm up
      frame_dropped_filter.x = 0.
      frames_dropped = 0.
    run_count = run_count + 1

    frame_drop_ratio = frames_dropped / (1 + frames_dropped)
    prepare_only = vipc_dropped_frames > 0
    if prepare_only:
      cloudlog.error(f"skipping model eval. Dropped {vipc_dropped_frames} frames")

    bufs = {name: buf_extra if 'big' in name else buf_main for name in model.vision_input_names}
    transforms = {name: model_transform_extra if 'big' in name else model_transform_main for name in model.vision_input_names}

    frame_delay = DT_MDL # compensate for time passed since the frame was captured: current_time - timestamp_eof is 50ms on average
    action_delay = DT_MDL / 2 # middle of the interval between model output (current state) and next frame (expected state)
    lat_action_t = lat_delay + frame_delay + action_delay
    long_action_t = long_delay + frame_delay + action_delay

    inputs:dict[str, np.ndarray] = {
      model.desire_key: vec_desire,
      'traffic_convention': traffic_convention,
    }

    if 'lateral_control_params' in model.numpy_inputs:
      inputs['lateral_control_params'] = np.array([v_ego, lat_delay], dtype=np.float32)

    if 'action_t' in model.numpy_inputs:
      inputs['action_t'] = np.array([lat_action_t, long_action_t], dtype=np.float32)

    mt1 = time.perf_counter()
    try:
      send_chestnut = (chestnut_state is not None and
                       run_count % round(model.constants.MODEL_FREQ / SERVICE_LIST['chestnutState'].frequency) == 0)
      model_output = model.run(bufs, transforms, inputs, prepare_only, chestnut_state.send if send_chestnut else None)
    except Exception:
      if not params.get_bool("ChestnutActive"):
        raise
      cloudlog.exception("chestnut failed, falling back to small")
      params.put_bool("ChestnutActive", False)
      assert small_model is not None
      model = small_model
      if chestnut_state is not None:
        chestnut_state.big = False
      run_count = 0
      model_output = None
    mt2 = time.perf_counter()
    model_execution_time = mt2 - mt1

    if model_output is not None:
      modelv2_send = messaging.new_message('modelV2')
      drivingdata_send = messaging.new_message('drivingModelData')
      posenet_send = messaging.new_message('cameraOdometry')
      mdv2sp_send = messaging.new_message('modelDataV2SP')

      action = model.get_action_from_model(model_output, prev_action, lat_action_t, long_action_t, v_ego)
      prev_action = action
      fill_model_msg(drivingdata_send, modelv2_send, model_output, action,
                     publish_state, meta_main.frame_id, meta_extra.frame_id, frame_id,
                     frame_drop_ratio, meta_main.timestamp_eof, model_execution_time, live_calib_seen, meta_constants)
      modelv2_send.modelV2.big = model.chestnut

      desire_state = modelv2_send.modelV2.meta.desireState
      l_lane_change_prob = desire_state[log.Desire.laneChangeLeft]
      r_lane_change_prob = desire_state[log.Desire.laneChangeRight]
      lane_change_prob = l_lane_change_prob + r_lane_change_prob
      left_edge, right_edge = RELC.update_and_fill(modelv2_send.modelV2, mdv2sp_send.modelDataV2SP, v_ego)
      DH.update(sm['carState'], sm['carControl'].latActive, lane_change_prob, left_edge, right_edge)
      modelv2_send.modelV2.meta.laneChangeState = DH.lane_change_state
      modelv2_send.modelV2.meta.laneChangeDirection = DH.lane_change_direction
      mdv2sp_send.modelDataV2SP.laneTurnDirection = DH.lane_turn_direction
      drivingdata_send.drivingModelData.meta.laneChangeState = DH.lane_change_state
      drivingdata_send.drivingModelData.meta.laneChangeDirection = DH.lane_change_direction

      fill_pose_msg(posenet_send, model_output, meta_main.frame_id, vipc_dropped_frames, meta_main.timestamp_eof, live_calib_seen)
      pm.send('modelV2', modelv2_send)
      pm.send('drivingModelData', drivingdata_send)
      pm.send('cameraOdometry', posenet_send)
      pm.send('modelDataV2SP', mdv2sp_send)
    last_vipc_frame_id = meta_main.frame_id

if __name__ == "__main__":
  try:
    capture_stdio()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--demo', action='store_true', help='A boolean for demo mode.')
    args = parser.parse_args()
    main(demo=args.demo)
  except KeyboardInterrupt:
    cloudlog.warning(f"child {PROCESS_NAME} got SIGINT")
  except Exception:
    sentry.capture_exception()
    # GS450h: eGPU で走っていたなら、死ぬ直前に USB を再列挙しておく。tinygrad の
    # "Wait timeout" は GPU がハングしたまま残るので、プロセスを殺し直すだけでは
    # 次のロードも失敗する。manager 側の再起動 (restart_on_crash) と対で効く。
    try:
      save_stdio_snapshot("-crash")
      save_dmesg_snapshot("-crash")
      if Params().get_bool("ChestnutActive"):
        reset_chestnut()
    except Exception:
      cloudlog.exception("reset_chestnut on crash failed")
    raise
