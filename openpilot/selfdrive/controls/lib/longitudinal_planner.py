#!/usr/bin/env python3
import numpy as np

import openpilot.cereal.messaging as messaging
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


def get_cruise_accel(v_cruise, v_ego, a_cruise_prev, dt):
  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, ACCEL_MAX)
  j_cruise = np.interp(v_ego, J_CRUISE_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


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

    self.a_cruise = get_cruise_accel(v_cruise, v_ego, self.a_cruise, self.dt)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    output_a_target_e2e = self.accel_boost.update(sm, output_a_target_e2e, output_a_target_mpc, self.a_cruise)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop),
                  (output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e)]

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
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
