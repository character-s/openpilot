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
from openpilot.sunnypilot.selfdrive.controls.lib.latcontrol_torque_ext_base import LatControlTorqueExtBase, sign
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.helpers import MOCK_MODEL_PATH
from openpilot.sunnypilot.selfdrive.controls.lib.nnlc.model import NNTorqueModel

LOW_SPEED_X = [0, 10, 20, 30]
LOW_SPEED_Y = [12, 3, 1, 0]

# GS 450h: 持続カーブでプラントゲインが上がり fulfill が入り 93% -> 3-5s 後 105% まで育つ
# (トルクは減っているのに曲がり続ける)。定常だけ FF を絞ってこの上振れを打ち消す。
# JSON 側で一様に 5% 絞る v6shrink は狙いどおり定常を 99.8% にしたが、入りまで 93.4->88.8% と
# 削って user 体感の切り込み不足を招いたため、時間依存のこちらへ移行 (2026-07-28)。
# 併用禁止: この trim を入れるときは model を v5f に戻すこと (v6shrink と二重に効く)。
SETTLED_LA_MIN = 0.5           # これ未満は「カーブ中」と見なさない
SETTLED_RAMP_X = [1.0, 2.5]    # 同符号カーブの継続時間 [s]
SETTLED_RAMP_Y = [0.0, 1.0]    # 補正の効き (0 = 無補正)

# ★ 07-28: 固定 5% から (v, |la|) テーブルへ。rlog 20 route / 1184 episode 窓の実測
# 残差面 (`archive/probes/_calib_residual_surface.py`) で、定常 fulfill が
#   29-61km/h: 107.2 / 107.1 / 103.7 / 96.7%   (|la| .5-.7 / .7-1.0 / 1.0-1.5 / 1.5+)
#   >61km/h  : 106.6 / 105.1 / 105.8 / 103.0%
#   18-29km/h: (薄) / -- / -- / 89.1%
# と **小 la で過剰・大 la で不足**と判明したため。固定 5% 減は大 la 帯 (交差点の切り込み)
# を悪化させていた (18-29 × la1.5+ は元々 11% 不足なのに更に 5% 削っていた)。
# 正 = FF を減らす / 負 = FF を増やす。trim = 1 - 1/fulfill。
# ★ 07-28 夜: ±3% 制限を実測値へ開いた (キャップ ±8%)。初回 ±3% 版で 1 走行 (f4-f6、52min)
# 判定した結果 — 低速 overshoot 156.7 -> 90.2 ep/h・29-61 帯 93.0 -> 35.5 ep/h と改善、
# 狭帯カーブ clear p05 は最良級 (f6 1.19m)、user 体感は「プラシーボ程度」= 副作用なし。
# 3% では体感にも残差面にも足りないので、採用セル (LOO 符号一致 + bootstrap 90%CI) の
# 実測値をそのまま入れる。18-29 × la1.5+ のみ実測 -12.2% を -8% にキャップ。
# n 不足セル (18-29 の .5-.7 / .7-1.0 / 1.0-1.5) は 29-61 行より控えめな外挿のまま。
SETTLED_TRIM_LA = [0.6, 0.85, 1.25, 2.0]      # |desired la| の代表点
SETTLED_TRIM_V = [7.0, 12.0, 20.0]            # vEgo [m/s] の代表点 (25 / 43 / 72 km/h)
SETTLED_TRIM_TBL = [
    [0.050, 0.050, 0.000, -0.080],            # 18-29km/h  (la1.5+ 実測 -0.122 を -0.08 へキャップ)
    [0.067, 0.066, 0.036, -0.034],            # 29-61km/h  (107.2 / 107.1 / 103.7 / 96.7%)
    [0.062, 0.049, 0.055,  0.029],            # >61km/h    (106.6 / 105.1 / 105.8 / 103.0%)
]

# ★ 07-31: e_y (車線中心復元) feedback。FF trim は追従の校正であって、平衡点が車線中心より
# 0.18-0.44m イン側に居る問題は動かせない (07-30 実車確認: trim ±8% でも「線を踏みそう/
# はみ出し傾向は不変」)。平衡点を動かせるのは位置フィードバックだけなので、車線中心からの
# ずれ e_y を setpoint に注入する。
# - 注入先は _setpoint のみ (desired 系は触らない)。FF/settled trim/残差面測定と完全分離、
#   「位置は error 経路の担当」の分業を守る。効きは NN 勾配 × PID (定常は I) 経由 = 時定数
#   数秒の遅いループ = 位置制御として適切な帯域。
# - パラメータは shadow スイープで選定 (archive/probes/_lane_analysis.py --only=feedback、
#   f4-ff 10 route / n=151,342 / 踏み 1,301 件)。K=0.6/dz=0.05 で 0.35m イン寄り時 0.18 m/s²
#   の打ち消し。踏み時正答 76.8% (逆符号に書くと 14% = 符号系はデータ実証済み)。
# - 生 offset の Δp95 0.023 が門値 0.02 を僅かに超えるため LPF (RC 0.3s) を挟む。
# - 中央線なし対面通行 (path が道路中央へ寄る場面) の gating は初版では入れない
#   (user 07-31: 様子見で必要なら)。prob/幅/速度の門が怪しい白線をある程度自然に切る。
#
# ★★ 07-31 実走で初版 (P 0.6 / limit 0.35 / deadzone のみ) の欠陥が 4 つ出たので全面改訂。
# 実測: e_y が効く帯だけ |offset| が減り (43-61km/h で 0.183→0.108m = -41%) 方向は正しい。
# だが同じ帯で fulfill が -4pt (29-61 95.6→92.0 / >61 100.1→95.6)、user 体感は
# 「カーブ最初の切り方が悪い、心臓に悪い、override につながる」。原因は 2 つ:
#  (1) limit 0.35 m/s^2 は位置制御として 1 桁過大。横位置を 0.3m 動かすのに要るのは
#      数秒スケールで 0.02-0.07 m/s^2。0.35 は la=1.0 のカーブの 35% を削る量で、
#      shadow の「踏み時 p50 0.28」= カーブでイン寄りになった瞬間に曲がりを 3 割削っていた。
#      → limit/K を 1/3.5 に。位置は P でなく I で動かす (下記)。
#  (2) 切り込み中も full gain で効いていた。e_y の目的は「平衡点の移動」なので過渡で
#      効かせる理由がない。進入だけでなく緩→急の複合カーブでも切り込みを妨げる
#      (user 実走報告) → 継続時間 ramp ではなく **jerk (曲率変化率) で gate** する。
#      lookahead_lateral_jerk は先読み値なので、gate が切り込みより先に立つ。
# 追加した I: P を下げた分、定常偏差は積分で埋める。planner のイン寄りは 07-28 の
# drift 回帰で「自車位置に依存しない一定量」と実測済み = 定数外乱 = I の守備範囲。
# 時定数は分オーダー (0.3m のずれが ~55s 続いて 0.05 m/s^2)。torqued の friction 推定と
# 同じ「ゆっくり効く」思想 (user 07-31 の提案)。gate 中と白線ロスト中は凍結 (ワインドアップ防止)。
# e_psi (減衰項) は次段。user 体感で振動は「サイドミラーで分かる程度」= 許容範囲、
# かつ P を 1/4 にすると振れ幅自体が縮むため。それでも残るなら laneLines の傾きから入れる。
EY_K = 0.15             # [m/s^2 per m] 復元ゲイン (P)。0.6 は切れ角を削りすぎた
EY_DEADZONE = 0.02      # [m] リレー的な出入りが「寄って戻って」を生むので縮小
EY_LIMIT = 0.10         # [m/s^2] P+I の合計上限。la=1.0 のカーブへの影響を 10% 以内に
EY_KI = 0.003           # [m/s^2 per m·s] 遅い積分。0.3m×55s で 0.05 m/s^2
EY_I_LIMIT = 0.08       # [m/s^2] I 単体の上限
EY_GATE_JERK = [0.3, 1.0]   # |lookahead lat jerk| [m/s^3]: 0.3 以下 = 定常 → full、1.0 以上 = 切り込み → 0
EY_V_MIN = 8.0          # [m/s] 低速は白線が視野から外れやすい (07-31 実測で通過率 45% vs 高速 74%)
EY_PROB_MIN = 0.5       # 両側 laneLineProb の門
EY_WIDTH_MIN, EY_WIDTH_MAX = 2.4, 4.2   # [m] 車線幅の妥当性門
EY_LPF_RC = 0.3         # [s]


# At a given roll, if pitch magnitude increases, the
# gravitational acceleration component starts pointing
# in the longitudinal direction, decreasing the lateral
# acceleration component. Here we do the same thing
# to the roll value itself, then passed to nnff.
def roll_pitch_adjust(roll, pitch):
  return roll * math.cos(pitch)


class NeuralNetworkLateralControl(LatControlTorqueExtBase):
  def __init__(self, lac_torque, CP, CP_SP, CI):
    super().__init__(lac_torque, CP, CP_SP, CI)
    self.params = Params()
    self.enabled = self.params.get_bool("NeuralNetworkLateralControl")
    self.has_nn_model = CP_SP.neuralNetworkLateralControl.model.path != MOCK_MODEL_PATH

    # NN model takes current v_ego, lateral_accel, lat accel/jerk error, roll, and past/future/planned data
    # of lat accel and roll
    # Past value is computed using previous desired lat accel and observed roll
    self.model = NNTorqueModel(CP_SP.neuralNetworkLateralControl.model.path)

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

    # e_y feedback の LPF (白線検出ノイズ/跳び対策) と遅い積分
    self.ey_filter = FirstOrderFilter(0.0, EY_LPF_RC, 0.01)
    self.ey_integral = 0.0

  @property
  def _nnlc_enabled(self):
    return self.enabled and self.model_valid and self.has_nn_model

  @property
  def output_pid(self):
    """★ NNL-8: 出力を実際に作った PID (NNLC 無効時は None = base 側)。rlog の p/i/d/f 用。

    NNL-8 で PID を base と分けたので、latcontrol_torque.py が自分の self.pid を
    そのまま logging すると出力を作っていない側の内訳が rlog に出てしまう。
    """
    return self._pid if self._nnlc_enabled else None

  def update_limits(self):
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
    self._pid_log.error = float(torque_from_setpoint - torque_from_measurement)
    self._ff = self.torque_from_lateral_accel_in_torque_space(LatControlInputs(self._gravity_adjusted_lateral_accel, self._roll_compensation,
                                                                               CS.vEgo, CS.aEgo), self.torque_params, gravity_adjusted=True)
    self._ff += get_friction_in_torque_space(self._desired_lateral_accel - self._actual_lateral_accel, self._lateral_accel_deadzone,
                                             FRICTION_THRESHOLD, self.torque_params)

  def update_output_torque(self, CS):
    self.update_limits()  # Stage 1 (A): set PID limits right before PID.update
    freeze_integrator = self._steer_limited_by_safety or CS.steeringPressed or CS.vEgo < 5
    self._output_torque = self._pid.update(self._pid_log.error,
                                           feedforward=self._ff,
                                           speed=CS.vEgo,
                                           freeze_integrator=freeze_integrator)

  def update_neural_network_feedforward(self, CS, params, calibrated_pose) -> None:
    if not self._nnlc_enabled:
      return

    self.update_feedforward_torque_space(CS)

    low_speed_factor = float(np.interp(CS.vEgo, LOW_SPEED_X, LOW_SPEED_Y)) ** 2
    self._setpoint = self._desired_lateral_accel + low_speed_factor * self._desired_curvature
    self._measurement = self._actual_lateral_accel + low_speed_factor * self._actual_curvature

    # e_y: 車線中心からのずれを setpoint に注入して平衡点を中心へ引き戻す。
    # offset = (yl+yr)/2、正 = 中心が右 (laneLines y は右正)。中心が右なら右へ寄せる = 正の la。
    # laneLines y 系と curvature/la 系が同符号規約であることは shadow の踏み正答率で実証済み。
    ey_corr, ey_off, ey_valid = 0.0, 0.0, False
    m2 = self.model_v2
    if m2 is not None and len(m2.laneLines) >= 3 and len(m2.laneLineProbs) >= 3:
      yl, yr = m2.laneLines[1].y, m2.laneLines[2].y
      if len(yl) and len(yr) and float(m2.laneLineProbs[1]) > EY_PROB_MIN and float(m2.laneLineProbs[2]) > EY_PROB_MIN:
        lane_w = float(yr[0]) - float(yl[0])
        if EY_WIDTH_MIN < lane_w < EY_WIDTH_MAX and CS.vEgo > EY_V_MIN:
          ey_off = (float(yl[0]) + float(yr[0])) / 2.0
          ey_valid = True

    # gate: 切り込み中 (曲率が変化中) は e_y を落とす。平衡点を動かすのが目的なので過渡で
    # 効かせる必要がなく、効かせると切り込みを妨げる (07-31 実走: 「突っ込んでから曲がる」)。
    # lookahead_lateral_jerk は先読み値 = gate が切り込みより先に立つ。継続時間ベースの
    # ramp では緩→急の複合カーブを守れないので、変化率で見るのが正しい。
    ey_gate = float(np.interp(abs(self.lookahead_lateral_jerk), EY_GATE_JERK, [1.0, 0.0]))

    # 遅い I: 定常偏差 (planner の位置非依存なイン寄り) を埋める担当。P では平衡点は動かない。
    # 切り込み中と白線ロスト中は凍結してワインドアップを防ぐ。
    if ey_valid and ey_gate > 0.5:
      self.ey_integral = float(np.clip(self.ey_integral + EY_KI * ey_off * 0.01,
                                       -EY_I_LIMIT, EY_I_LIMIT))
    elif not ey_valid:
      self.ey_integral *= 0.999   # 白線が長く取れない道では徐々に忘れる

    if ey_valid:
      ey_p = sign(ey_off) * EY_K * max(abs(ey_off) - EY_DEADZONE, 0.0)
      ey_corr = float(np.clip(ey_p + self.ey_integral, -EY_LIMIT, EY_LIMIT)) * ey_gate
    self._setpoint += self.ey_filter.update(ey_corr)

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
    self._pid_log.error = torque_from_setpoint - torque_from_measurement

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
        self._pid_log.error = self._pid_log.error * (1.0 - error_blend_factor) + torque_from_error * error_blend_factor

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
