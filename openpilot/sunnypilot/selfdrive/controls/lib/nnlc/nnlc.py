"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import deque
import math
import numpy as np

from opendbc.car.lateral import FRICTION_THRESHOLD, get_friction
from opendbc.sunnypilot.car.interfaces import LatControlInputs
from opendbc.sunnypilot.car.lateral_ext import get_friction as get_friction_in_torque_space
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.params import Params
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import sign
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_jerk_aware import LatControlTorqueJerkAware
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]

# GS 450h settled FF trim: 持続カーブでプラントゲインが上がり、定常で fulfill が育つ
# (小 |la| で過剰・大 |la| で不足 = 固定 % では駄目)。入り (turn-in) は無補正で定常だけ補正する。
# trim = 1 - 1/fulfill (正 = FF を減らす / 負 = 増やす)。|trim| は 8% でキャップ。
# 併用禁止: JSON 側で一様に絞る v6shrink とは二重に効く (model は v5f のまま使う)。
#
# ★★ 2026-09-20 改訂 (根拠 = archive/probes/_lat_postdrive.py の §8c / §8f、CTMV2 298min):
#   ① ランプを [1.0, 2.5] -> [0.5, 1.5] へ短縮。旧値では **ramp=1 の滞在率が 0-6% (>61km/h でも
#      15-30%)、実効 trim の中央値がほぼ +0.0%** = テーブルが滞在時間の 98% に届いていなかった。
#      2.5s 未満で抜ける曲がり角には事実上 1 bit も掛かっていない。
#      ⇒ 短縮で届く率 1.8% -> 2.8%、>61km/h の overshoot -16.7%、fulfill p50 103.7 -> 101.3%。
#        **rate limit / EPS 上限はどちらも ±0** = 出す側の余力を一切食わない。
#      ⚠⚠ **[0.3, 1.0] まで詰めない**。overshoot は -27.8% / fulfill 100.4% まで伸びるが、
#        **入り (turn-in) への干渉が +0.77% -> +1.81% と倍増する** (>61km/h、|des la|>0.5)。
#        settled trim の設計意図は「入りは無補正で定常だけ補正する」で、**旧設定では全帯で
#        切り込み中の実効 trim が +0.00%** だった。>61km/h の turn-in fulfill は既に
#        **99.4-101.9% = ちょうど**なので、そこを 97.6-100% まで削る意味がない。
#        ⇒ **絞りたいのは定常側**なので [0.5, 1.5] で止める。効果不足なら実走判定の後で詰める。
#      ⚠ [0.0, 0.3] は論外 (全体 fulfill が 98.1% = 足りない側へ行き過ぎる)。
#   ② 低中速 (V=7 / 12 の行) の「絞る」値を 0 にした。**実測は全セル gain<1 = 足りない側**で、
#      旧値は真逆を向いていた (43km/h x |la|0.85 で gain 0.967 なのに +6.6% 絞る)。
#      旧テーブルは 08-18 に 16D 期の desired で測ったもので、big が IDM/CTM/CTMV2 と替わって
#      この帯が反転した。届いていなかったので実害が出ていなかっただけ。
#      ⚠⚠ **実測どおり「増やす」側 (-0.01〜-0.08) には振っていない** — shadow (ランプ込み) で
#      低中速を持ち上げると overshoot が <29km/h +9.5% / 29-61 +20.8% と悪化するため。
#      ⇒ **まず「間違った方向に絞るのをやめる」だけに留める**。増やすかは実走判定の後。
#   ③ >61km/h は実測に合わせて絞り量を弱めた (0.062/0.049/0.055 -> 0.025/0.044/0.036)。
#      ⚠ >61km/h x |la|2.0 は n=11s と薄いので旧値 (+0.029) を据え置き。
#   ⚠ 急カーブ (|la|>=1.5) が足りないのは **FF ではなくモデル層** と 09-20 に決着している
#     (制御は desired より 169ms 先出し / rate limit の遅れは p50 ゼロ) ⇒ ここを増やしても直らない。
#
# ★★★ 2026-09-22 改訂 (根拠 = _lane_analysis.py --only=drift、CTMV2 546min / 1006 episode):
#   user 体感「イン食いが多い」を実測で裏付けた。**閉ループの安定平衡点が車線中心よりイン側**:
#     幅 <2.9m +0.07m / 2.9-3.1 +0.05m / 3.1-3.3 +0.14m / >3.3m +0.22m
#     曲率帯別 = 緩 R>250 0.148m / **R125-250 0.210m (最悪)** / 急 R<63 0.133m
#   ⚠ 「走るうちに寄っていく」のではない (drift rate p50 +0.004 m/s = 寄らない側) =
#     **最初から寄った位置で安定する平衡点型**。
#   ★ **速度別に割れる**: 29-61km/h の drift p25 **-0.045** に対し >61km/h は **-0.009**
#     = 高速ほど寄りにくい。**72km/h 行だけ trim が入っている (+2.5〜4.4%) のと符合する**
#     ⇒ **43km/h 行が 0.000 なのがイン食いを放置している**、というのがこの改訂の仮説。
#   ★ 必要量は極小: 復元ゲイン a=-0.265/s (episode 長補正後、時定数 3.8s) から
#     **平衡点を Δ 動かして保持するのに要る la = a^2*Δ = 0.0320*Δ** ⇒ Δ=0.14m で 0.0045 m/s^2。
#     ランプ (0.5-1.5s) と時定数の取りこぼしで実効 1.7 倍とみて 0.0077 m/s^2、
#     la=0.85 のカーブなら **trim +0.009** ⇒ 切りよく **+0.010 (1%)** を 0.85 / 1.25 セルに入れる。
#     (R125-250 を 43km/h で回ると la = v^2/R = 0.8 なので、効かせたい帯はこの 2 セル)
#   ⚠⚠ **これは §8f の再フィット (fulfill ベース) と真逆を向く**。8f は 43km/h を
#     実測 g=0.922-0.982 = 「足りない ⇒ 増やせ (負)」と推奨している。
#     **曲率の追従 (fulfill) と車線内位置 (イン食い) が逆方向を要求している**ためで、
#     イン食いを取るなら fulfill 1-2pt の低下は織り込む必要がある (29-61 定常は今 97.1%)。
#     ⇒ **実走判定で「イン食いは減ったか」と「fulfill は落ちすぎていないか」を必ず並べて見る**。
#   ⏭ 判定 = `_lane_analysis.py --only=drift` の幅帯別平衡点 (3.1-3.3 の +0.14m が縮むか) と
#     `_lat_postdrive.py` の §9a 29-61 定常 fulfill (97.1% からの低下幅)。
SETTLED_LA_MIN = 0.5           # これ未満は「カーブ中」と見なさない
SETTLED_RAMP_X = [0.5, 1.5]    # 同符号カーブの継続時間 [s] (09-20: [1.0, 2.5] から短縮)
SETTLED_RAMP_Y = [0.0, 1.0]    # 補正の効き (0 = 無補正)

SETTLED_TRIM_LA = [0.6, 0.85, 1.25, 2.0]      # |desired la| の代表点
SETTLED_TRIM_V = [7.0, 12.0, 20.0]            # vEgo [m/s] の代表点 (25 / 43 / 72 km/h)
SETTLED_TRIM_TBL = [
    # 実測 gain (09-20, ramp=0) を併記。正 = 絞る / 負 = 増やす。
    [0.000, 0.000, 0.000, -0.080],            # 25km/h  実測 g=0.960/0.967/0.882/0.879 (全部 1 未満 = 絞らない)
    [0.000, 0.010, 0.010, -0.034],            # 43km/h  09-22: 0.85/1.25 に +1.0% (イン食い。上の ★★★ 節)
    [0.025, 0.044, 0.036,  0.029],            # 72km/h  実測 g=1.026/1.046/1.037/(n不足で据え置き)
]


# At a given roll, if pitch magnitude increases, the
# gravitational acceleration component starts pointing
# in the longitudinal direction, decreasing the lateral
# acceleration component. Here we do the same thing
# to the roll value itself, then passed to nnff.
def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl(LatControlTorqueJerkAware):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    super().__init__(lac_torque, CP, CP_SP, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    model_path = CP_SP.neuralNetworkLateralControl.model.path
    self.has_nn_model = model_path not in (MOCK_MODEL_PATH, '')

    # NN model takes current v_ego, lateral_accel, lat accel/jerk error, roll, and past/future/planned data
    # of lat accel and roll
    # Past value is computed using previous desired lat accel and observed roll
    self.model = NNTorqueModel(model_path) if self.has_nn_model else None

    self.pitch = FirstOrderFilter(0.0, 0.5, 0.01)
    self.pitch_last = 0.0

    # setup future time offsets
    self.future_times = [0.3, 0.6, 1.0, 1.5] # seconds in the future
    self.nn_future_times = [i + self.desired_lat_jerk_time for i in self.future_times]

    # setup past time offsets
    self.past_times = [-0.3, -0.2, -0.1]
    history_check_frames = [int(abs(i)*100) for i in self.past_times]
    self.history_frame_offsets = [history_check_frames[0] - i for i in history_check_frames]
    self.lateral_accel_desired_deque = deque(maxlen=history_check_frames[0])
    self.roll_deque = deque(maxlen=history_check_frames[0])
    self.error_deque = deque(maxlen=history_check_frames[0])
    self.past_future_len = len(self.past_times) + len(self.nn_future_times)

    # settled FF trim の状態 (同符号カーブの継続時間と符号)
    self._settled_time = 0.0
    self._settled_sign = 0.0

  @property
  def _nnlc_enabled(self):
    return self.enabled and self.model_valid and self.has_nn_model

  @property
  def output_pid(self):
    """NNL-8: 出力を実際に作った PID (NNLC 無効時は None = base 側)。

    PID を base と分けたので、latcontrol_torque.py の p/i/d/f logging はこちらを見る必要がある。
    """
    return self._pid if self._nnlc_enabled else None

  def update_limits(self):
    super().update_limits()
    if not self._nnlc_enabled:
      return

    self._pid.set_limits(self.lac_torque.steer_max, -self.lac_torque.steer_max)

  def update_lateral_lag(self, lag):
    super().update_lateral_lag(lag)
    self.nn_future_times = [t + self.desired_lat_jerk_time for t in self.future_times]

  def update_feedforward_torque_space(self, CS):
    torque_from_setpoint = self.torque_from_lateral_accel_in_torque_space(LatControlInputs(self._setpoint, self._roll_compensation, CS.vEgo, CS.aEgo),
                                                                          self.torque_params, gravity_adjusted=False)
    torque_from_measurement = self.torque_from_lateral_accel_in_torque_space(LatControlInputs(self._measurement, self._roll_compensation, CS.vEgo, CS.aEgo),
                                                                             self.torque_params, gravity_adjusted=False)
    self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)  # ty: ignore[invalid-assignment]
    self._ff = self.torque_from_lateral_accel_in_torque_space(LatControlInputs(self._gravity_adjusted_lateral_accel, self._roll_compensation,
                                                                               CS.vEgo, CS.aEgo), self.torque_params, gravity_adjusted=True)
    self._ff += get_friction_in_torque_space(self._desired_lateral_accel - self._actual_lateral_accel, self._lateral_accel_deadzone,
                                             FRICTION_THRESHOLD, self.torque_params)

  def update_output_torque(self, CS):
    self.update_limits()  # Stage 1 (A): set PID limits right before PID.update
    super().update_output_torque(CS)

  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    if not self._nnlc_enabled:
      return

    self.update_feedforward_torque_space(CS)

    low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
    self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature

    # update past data
    roll = params.roll
    if calibrated_pose is not None:
      pitch = self.pitch.update(calibrated_pose.orientation.pitch)
      roll = roll_pitch_adjust(roll, pitch)
      self.pitch_last = pitch
    self.roll_deque.append(roll)
    self.lateral_accel_desired_deque.append(self._desired_lateral_accel)

    # prepare past and future values
    # adjust future times to account for longitudinal acceleration
    adjusted_future_times = [t + 0.5 * CS.aEgo * (t / max(CS.vEgo, 1.0)) for t in self.nn_future_times]
    past_rolls = [self.roll_deque[min(len(self.roll_deque) - 1, i)] for i in self.history_frame_offsets]
    future_rolls = [roll_pitch_adjust(np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.x) + roll,
                                      np.interp(t, ModelConstants.T_IDXS, self.model_v2.orientation.y) + self.pitch_last) for t in
                    adjusted_future_times]
    past_lateral_accels_desired = [self.lateral_accel_desired_deque[min(len(self.lateral_accel_desired_deque) - 1, i)]
                                   for i in self.history_frame_offsets]
    future_planned_lateral_accels = [np.interp(t, ModelConstants.T_IDXS, self.model_v2.acceleration.y) for t in
                                     adjusted_future_times]

    # compute NNFF error response
    nnff_setpoint_input = [CS.vEgo, self._setpoint, self.lateral_jerk_setpoint, roll] \
                          + [self._setpoint] * self.past_future_len \
                          + past_rolls + future_rolls
    # past lateral accel error shouldn't count, so use past desired like the setpoint input
    nnff_measurement_input = [CS.vEgo, self._measurement, self.lateral_jerk_measurement, roll] \
                             + [self._measurement] * self.past_future_len \
                             + past_rolls + future_rolls
    torque_from_setpoint = self.model.evaluate(nnff_setpoint_input)
    torque_from_measurement = self.model.evaluate(nnff_measurement_input)
    self._pid_log.error = torque_from_setpoint - torque_from_measurement  # ty: ignore[invalid-assignment]

    # The "pure" NNLC error response can be too weak for cars whose models were trained
    # with a lack of high-magnitude lateral acceleration data, for which the NNLC model
    # torque response flattens out at high lateral accelerations.
    # This workaround blends in a guaranteed stronger error response only when the
    # desired lateral acceleration is high enough to warrant it, by using the lateral acceleration
    # error as the input to the NNLC model. This is not ideal, and potentially degrades the NNLC
    # accuracy for cars that don't have this issue, but it's necessary until a better NNLC model
    # structure is used that doesn't create this issue when high-magnitude data is missing.
    error_blend_factor = float(np.interp(abs(self._desired_lateral_accel), [1.0, 2.0], [0.0, 1.0]))
    if error_blend_factor > 0.0:  # blend in stronger error response when in high lat accel
      # NNFF inputs 5+ are optional, and if left out are replaced with 0.0 inside the NNFF class
      nnff_error_input = [CS.vEgo, self._setpoint - self._measurement, self.lateral_jerk_setpoint - self.lateral_jerk_measurement, 0.0]
      torque_from_error = self.model.evaluate(nnff_error_input)
      if sign(self._pid_log.error) == sign(torque_from_error) and abs(self._pid_log.error) < abs(torque_from_error):
        self._pid_log.error = self._pid_log.error * (1.0 - error_blend_factor) + torque_from_error * error_blend_factor  # ty: ignore[invalid-assignment]

    # compute feedforward (same as nn setpoint output)
    friction_input = self.update_friction_input(self._setpoint, self._measurement)
    nn_input = [CS.vEgo, self._desired_lateral_accel, friction_input, roll] \
               + past_lateral_accels_desired + future_planned_lateral_accels \
               + past_rolls + future_rolls
    self._ff = self.model.evaluate(nn_input)

    # settled (定常) 区間だけ FF を絞る。入り (turn-in) は無補正なので切り込みは落ちない。
    la = self._desired_lateral_accel
    if abs(la) > SETTLED_LA_MIN and (self._settled_sign == 0.0 or sign(la) == self._settled_sign):
      self._settled_sign = sign(la)
      self._settled_time += 0.01
    else:
      self._settled_time = 0.0
      self._settled_sign = 0.0
    # (v, |la|) の 2 段線形補間。格子点の間は連続に変化する。
    trim_rows = [float(np.interp(abs(la), SETTLED_TRIM_LA, row)) for row in SETTLED_TRIM_TBL]
    trim = float(np.interp(CS.vEgo, SETTLED_TRIM_V, trim_rows))
    self._ff *= 1.0 - trim * float(np.interp(self._settled_time, SETTLED_RAMP_X, SETTLED_RAMP_Y))

    # apply friction override for cars with low NN friction response
    if self.model.friction_override:
      self._pid_log.error += get_friction(friction_input, self._lateral_accel_deadzone, FRICTION_THRESHOLD, self.torque_params)

    self.update_output_torque(CS)
