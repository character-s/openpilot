#!/usr/bin/env python3
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.accel_boost import AccelBoost
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP

J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
J_CRUISE_BP = [0., 10.0, 25., 40.]
# PLN-5 (2026-09-06): 巡航の減速要求は摩擦ブレーキに入らない深さまでにする (user 判断)。
# a_cruise = clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel) は設定速度との差 (m/s) をそのまま
# 加速度 (m/s^2) にするので、-1.2 だと設定を 10km/h 下げただけで即 -1.2 が立つ。GS 450h の回生では出ない
# 深さなので摩擦ブレーキが入り、後続から見てブレーキランプが点く。
# 実測 (09-05 route 045 の 9 seg、archive/probes/_acc_brake_threshold.py):
#   PCM_CRUISE(466) の ACC_BRAKING は aTarget が -0.5 より深い全帯で点灯率 1.000、-0.3 より浅いと
#   0.03 以下。UN R13-H は 0.7 m/s^2 以下での制動灯点灯を禁止しているので -0.5 なら灯火には
#   出ない見込み (⚠ 実車での目視確認は未実施。点いていたら -0.3 に落とす)。
# ⚠ 本当に制動が要る場面 = 前車追従 (MPC 候補) と e2e 候補はこの定数を通らないので影響しない。
#   下り坂の速度超過もここに含まれるが、user 判断で「ブレーキを踏むほどではない」= -0.5 のまま。
# (10-19 追従: upstream も #39060 で同じ -0.5 にした)
A_CRUISE_MIN = -0.5
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
# PLN-1_8 (2026-08-29): e2e の action ヘッドは -2.7 付近で飽和する。route 016 の信号停止では
# 同じモデルの経路 (modelV2.acceleration) が -3.3 を描いているのに action は -2.67 止まりで、
# 止まりきれずにドライバーがブレーキを踏んだ。経路が -3.0..-3.5 を描いた 51 フレームは
# action が全部 -2.6x に潰れており、分布の裾ではなく飽和。壁はモデルの中にあって外からは
# 動かせないので、**同じモデルの経路**を持ってきて min() を取り、床で開け幅を決める。
#
# ⚠ 定数オフセットやスケールでは代用できない。壁に当たっていない route (00e、走行 11377
#   フレーム) で経路方式は 0 件なのに、一律 -0.4 は 2 件・1.25 倍も 2 件発火する。
#   **床を付けても減らない** — 誤発火は -2.10 -> -2.50 で床 -3.0 に届かないため。
#   床は「どこまで深くするか」の摘みで、「いつ発火するか」には関与しない。門になるのは
#   min() の方で、経路が action より浅ければ自動的に何も起きない。
# ⚠ modeld ではなく planner に置く。modeld で潰すと modelV2.action の記録値が補正後になり、
#   壁の位置をログから測り続けられなくなる (計測 = archive/probes/_e2e_accel_ceiling.py)。
E2E_ACCEL_PATH_GATE = -2.0   # これより深い要求のときだけ経路を見る
E2E_ACCEL_PATH_FLOOR = -3.0  # ここより深くはしない (ACCEL_MIN -3.5 に 0.5 残す)
E2E_ACCEL_PATH_N = 9         # 経路の先頭 0.6s (ModelConstants.T_IDXS[8] = 0.625)
E2E_ACCEL_PATH_RC = 0.15     # 経路 min の平滑 (modeld の LONG_SMOOTH_SECONDS と同じ桁)


# GS450h PLN-9 (10-19, user 判断「B」): 上流 0.11.3 の Autospeed は e2e を常に min の候補に入れるので、lead なしの加速が
# モデル任せになる。実測 (route 0f6): lead なし加速の 95% を e2e が min で取り、36-54km/h では設定速度まで 17km/h 余裕が
# あっても aTarget p50 0.09 m/s^2 (旧 GS の ACC 側は同じ帯で約 1.0)。⇒ e2e は「減速したい」ときだけ候補に入れ、
# 加速は巡航候補で決める (= 追従前の DEC の ACC モードに近い)。巡航候補の上限も旧版の速度別テーブルに戻す。
# ⚠ モデルを替えたら見直す (user 10-19「モデル変わったらまた考える」)。
A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]   # 旧版 (追従前) の巡航加速上限
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
MORE_AGGRESSIVE_ACCEL_SCALE = 1.25         # more aggressive だけ 1.25 倍 (10-10、ACCEL_MAX で頭打ち)
_A_TOTAL_MAX_V = [1.7, 3.2]                # カーブ中は横加速度と合わせた総加速度で抑える (旧版)
_A_TOTAL_MAX_BP = [20., 40.]
E2E_LOW_V = 5.0              # [m/s] 18km/h 未満は、計画が加速を描いていない限り e2e を候補に残す (停止・発進は e2e の止まれ
                             #   判定が要。外すと前車なしの赤信号で巡航候補が加速を出す)
E2E_LOW_GO_MARGIN = 0.3      # [m/s] PLN-11b: 計画速度 (T_IDXS[PLAN_GO_IDX] ~2.2s 先) が v_ego + これを超えたら低速でも e2e を外す。
                             #   10-19 は 18km/h 未満で e2e を常に残していたため、発進後 ~25km/h まで (18km/h + 保持 1s)
                             #   モデルの弱い加速が min を取っていた (user 10-20「25k あたりまでいくと強くなる」)
E2E_DECEL_GATE = -0.3        # [m/s^2] e2e がこれより強く減速を求めたら候補に入れる
E2E_SLOWDOWN_N = 21          # modelV2.velocity の先頭 ~4.4s (T_IDXS[20] = 3.9s, [21] = 4.3s)
E2E_SLOWDOWN_FRAC = 0.85     # 計画速度の最小がこれ x v_ego を下回る = モデルは減速するつもり
E2E_GATE_HOLD = 1.0          # [s] 一度入れたら最低これだけ保持 (わずかな出入りで指令が跳ねないように)

# GS450h PLN-10 (10-19, user 判断「L2 なので有効でよい」): lead なしの停止から自動で出ない = 青でも action ヘッドが
# shouldStop / accel ~0 のまま (08-20 以降 big の lead なし停止 81 回中 34 回がドライバー発進)。同じモデルの計画速度が
# 動いたら e2e の止まれを外す。持続の確認は longcontrol の PLN-3 (止まれが 1.5s 解け続けたら発進) に任せる。
# 事前評価 (_launch_accel_probe.py --go-rules、08-20 以降 204 route / 81 停止、持続 1.5s、20Hz 判定):
#   0.1 m/s = 早く出られる 57 / ⚠ 赤で許可 2 (0dc・0d8、fcam で赤を確認) / 救えない 22。user 10-19「偽陽性は許容する」で
#   救える回数が最大の値。計画の最遠点位置 (GreenLightAlert と同じ 30m) で判定する案は 48 / 4 / 29 で劣る
PLAN_GO_V = 0.1              # [m/s] modelV2.velocity の T_IDXS[15] (~2.2s 先) がこれを超えたら
PLAN_GO_IDX = 15             # ⚠ 1 点は 20Hz で揺れ (route ed は 0.08-0.38)、1 フレーム割ると PLN-3 の 1.5s がリセットされる
                             #   = ed は救えない。区間の最大 (1.0-3.9s 先) で見ると揺れは消えるが、赤のうちに遠い側が上がって
                             #   赤の候補が 17-29 回に増える (20Hz 評価) ので不採用
PLAN_GO_MAX_V = 3.0          # [m/s] ~11km/h 未満 (停止・発進直後) だけ
PLAN_GO_ACCEL = 0.5          # [m/s^2] 発進の加速度の下限 (穏やかに)

# GS450h PLN-11c (10-20): lead なしで e2e が減速を出すときも、止まるつもりでなければ PLN-5 の A_CRUISE_MIN (-0.5、制動灯が
# 点かない深さ) で止める。追従前は lead なしの減速が巡航候補だけだったので -0.5 で収まっていたが、PLN-9 で e2e の減速を
# そのまま候補に入れたため、lead なしの自主減速が強くなった (user 10-20「前はブレーキランプがつかない -0.5 までに制限してた」)。
# 例外 = 止まるつもり (action の止まれ / 計画速度が地平線内で E2E_STOP_PLAN_V を下回る = 信号・一時停止・飛び出し) と、
# 急な減速 (PLN-1_8 の門 E2E_ACCEL_PATH_GATE より深い)。⚠ 遠い赤信号への接近は、計画が止まりを描くまで -0.5 で減速する
E2E_STOP_PLAN_V = 1.0        # [m/s] modelV2.velocity (~10s 先まで) の最小がこれ未満 = モデルは止まるつもり


def get_max_accel(v_ego, personality=log.LongitudinalPersonality.standard):
  max_accel = float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))
  if personality == log.LongitudinalPersonality.moreAggressive:
    max_accel = min(max_accel * MORE_AGGRESSIVE_ACCEL_SCALE, ACCEL_MAX)
  return max_accel


def get_cruise_target(v_cruise, v_ego, personality=log.LongitudinalPersonality.standard, curvature=0.0):
  """巡航候補の目標 (ジャーク制限の前)。"""
  max_accel = get_max_accel(v_ego, personality)
  # カーブ中は総加速度 (前後 + 横) で抑える。横加速度は controlsState.curvature (舵角からオフセット補正済み) から
  a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
  a_y = v_ego ** 2 * abs(curvature)
  max_accel = min(max_accel, float(np.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))))
  return float(np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel))


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt, personality=log.LongitudinalPersonality.standard, curvature=0.0):
  target_accel = get_cruise_target(v_cruise, v_ego, personality, curvature)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


def e2e_wants_decel(v_ego, e2e_accel, e2e_should_stop, plan_v):
  """PLN-9: e2e を min の候補に入れるか (= モデルが減速・停止したい場面か)。"""
  if e2e_should_stop or e2e_accel < E2E_DECEL_GATE:
    return True
  if v_ego < E2E_LOW_V:
    # 低速: 計画が加速を描いているときだけ外す (止まれは上で拾う = 止まれが立っている間は外れない)
    return not (len(plan_v) > PLAN_GO_IDX and plan_v[PLAN_GO_IDX] > v_ego + E2E_LOW_GO_MARGIN)
  if len(plan_v) >= E2E_SLOWDOWN_N:
    return float(np.min(np.array(plan_v)[:E2E_SLOWDOWN_N])) < E2E_SLOWDOWN_FRAC * v_ego
  return True  # 計画が読めないときは上流どおり (安全側)


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.accel_boost = AccelBoost()

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.e2e_path_accel = FirstOrderFilter(0.0, E2E_ACCEL_PATH_RC, self.dt)  # PLN-1_8
    self.e2e_gate_timer = 0.0  # PLN-9: e2e を候補に入れ続ける残り時間 [s]
    self.last_plan_source = LongitudinalPlanSource.cruise  # PLN-9: 前回 min を取った候補
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    if self.CP.openpilotLongitudinalControl:
      self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    else:
      self.mpc.set_cur_state(v_ego, sm['carState'].aEgo)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    # PLN-1_8: action ヘッドの飽和を、同じモデルの経路で補う。⚠ 門が閉じている間もフィルタは
    # 回す (開いた瞬間に古い値を出さないため)。shouldStop はモデルの判断のまま触らない。
    # ⚠ capnp のリストは直接スライスしない。この repo の作法は np.array() 経由
    #   (sunnypilot/.../vision_controller.py と同じ)。
    # ⚠ 10-19 追従: accel_boost より前に置く (boost は補正後の e2e 候補に乗る)。
    accel_path = sm['modelV2'].acceleration.x
    if len(accel_path) >= E2E_ACCEL_PATH_N:
      path_accel = float(np.min(np.array(accel_path)[:E2E_ACCEL_PATH_N]))
      self.e2e_path_accel.update(float(np.clip(path_accel, ACCEL_MIN, ACCEL_MAX)))
      if output_a_target_e2e < E2E_ACCEL_PATH_GATE:
        output_a_target_e2e = max(min(output_a_target_e2e, self.e2e_path_accel.x), E2E_ACCEL_PATH_FLOOR)

    # PLN-10: lead なしの停止で、計画は動いたのに action が止まれのまま → e2e の止まれを外して穏やかに出す
    plan_v = sm['modelV2'].velocity.x
    if (v_ego < PLAN_GO_MAX_V and not sm['radarState'].leadOne.present and sm['modelV2'].big
            and len(plan_v) > PLAN_GO_IDX and plan_v[PLAN_GO_IDX] > PLAN_GO_V):
      output_should_stop_e2e = False
      output_a_target_e2e = max(output_a_target_e2e, PLAN_GO_ACCEL)

    # PLN-11c: lead なしで止まるつもりのない e2e 減速は -0.5 (制動灯が点かない深さ) まで
    if (not sm['radarState'].leadOne.present and not output_should_stop_e2e and output_a_target_e2e > E2E_ACCEL_PATH_GATE
            and len(plan_v) and float(np.min(np.array(plan_v))) > E2E_STOP_PLAN_V):
      output_a_target_e2e = max(output_a_target_e2e, A_CRUISE_MIN)

    # PLN-9: e2e は減速・停止したい場面だけ min の候補にする (上流は常に候補)。判定は boost を足す前のモデルの値で
    if e2e_wants_decel(v_ego, output_a_target_e2e, output_should_stop_e2e, sm['modelV2'].velocity.x):
      self.e2e_gate_timer = E2E_GATE_HOLD
    else:
      self.e2e_gate_timer = max(self.e2e_gate_timer - self.dt, 0.0)
    use_e2e = self.e2e_gate_timer > 0.0

    # PLN-9: 巡航候補は前回 min を取っていなかったなら今の指令から立ち上げる (e2e -> 巡航の切替で指令が跳ねないように)。
    # ⚠ self.mpc.source は上の mpc.update() で上書きされるので、前回の選択は self.last_plan_source で持つ
    a_cruise_prev = self.a_cruise if self.last_plan_source == LongitudinalPlanSource.cruise else min(self.a_cruise, a_prev)
    self.a_cruise = get_cruise_accel(v_cruise, v_ego, a_cruise_prev, self.dt,
                                     sm['selfdriveState'].personality, sm['controlsState'].curvature)
    # PLN-11a: 止まれは立ち上げ前の目標で判定する。停止中は指令 (e2e の止まれ) から立ち上げるので self.a_cruise も
    # 0.1 未満にいて、PLN-10 が止まれを外しても巡航候補が止まれを出し続け、PLN-3 の 1.5s の始まりが遅れていた
    # (user 10-20「lead なし発車の反応が前より遅い」)。上流の巡航候補は停止中も上限にいるので止まれを出さない。
    # forceDecel (v_cruise = 0) では目標が ~0 になり、今までどおり止まれを出す
    cruise_should_stop = should_stop(v_ego, get_cruise_target(v_cruise, v_ego, sm['selfdriveState'].personality,
                                                              sm['controlsState'].curvature))

    output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if use_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.last_plan_source = self.mpc.source
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.accelBoost = float(self.accel_boost.total_boost)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
