import numpy as np
from opendbc.car.structs import car
from openpilot.common.params import Params
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# PLN-3: resume hysteresis (kills lead-less creep oscillation at red lights)
PLN3_GO_SUSTAIN_LEAD = 0.5     # [s] shouldStop must stay false this long (lead present)
PLN3_LEAD_SUSTAIN = 1.0        # [s] hasLead must persist this long to count (ghost leads)

# Lead-less auto resume (pull away from a red light on its own), gated on which model is driving:
#   small (16D): None = disabled (measured false GO 3/9 while stopped)
#   big  (TT):   1.5s. 2.0 felt too slow; every real auto launch sat at go_timer 2.0x and the longest
#                aborted GO burst was 1.35s, so 1.5 launches the same ones and no flyer.
#                ⚠ only 0.15s of margin: if a false GO ever launches the car, put this back UP.
# ⚠ modeld falls back to small when the eGPU is missing/fails, so this follows the model at runtime.
PLN3_GO_SUSTAIN_NOLEAD = None      # [s] small model; None disables auto resume
PLN3_GO_SUSTAIN_NOLEAD_BIG = 1.5   # [s] big model on the eGPU


def long_control_state_trans(CP_SP, active, long_control_state,
                             should_stop, brake_pressed, cruise_standstill,
                             resume_ok=True):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and resume_ok:  # PLN-3
        long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.pid:
      if should_stop:
        long_control_state = LongCtrlState.stopping

  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP, params=None):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController(0.0, (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    self.go_timer = 0.0    # PLN-3: seconds shouldStop has been continuously false
    self.lead_timer = 0.0  # PLN-3: seconds hasLead has been continuously true
    # PLN-3: which model is driving. ChestnutActive is cleared on manager start / ignition and is
    # only set true once the big model has actually loaded on the eGPU, so "not yet loaded" and
    # "no eGPU at all" both read false = the conservative side.
    self._params = params or Params()
    self._frame = 0
    self._big_model = False

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, has_lead=False):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # PLN-3: track GO persistence and lead persistence at DT_CTRL rate
    self.go_timer = 0.0 if should_stop else min(self.go_timer + DT_CTRL, 10.0)
    self.lead_timer = min(self.lead_timer + DT_CTRL, 10.0) if has_lead else 0.0

    # PLN-3: refresh the model gate at 1Hz (same cadence dec.py reads its param at). The big
    # model takes ~80s to load, so this flips mid-drive - reading it once at init would miss it.
    self._frame += 1
    if self._frame % int(1. / DT_CTRL) == 0:
      self._big_model = self._params.get_bool("ChestnutActive")

    nolead_sustain = PLN3_GO_SUSTAIN_NOLEAD_BIG if self._big_model else PLN3_GO_SUSTAIN_NOLEAD
    go_sustain = PLN3_GO_SUSTAIN_LEAD if self.lead_timer >= PLN3_LEAD_SUSTAIN else nolead_sustain
    resume_ok = go_sustain is not None and self.go_timer >= go_sustain
    # escape: the car is already moving (driver launched with gas) - release the
    # hold so the stopping state does not drag the brake against them
    if CS.vEgo > self.CP.deprecated.vEgoStarting and not should_stop:
      resume_ok = True

    self.long_control_state = long_control_state_trans(self.CP_SP, active, self.long_control_state,
                                                       should_stop, CS.brakePressed,
                                                       CS.cruiseState.standstill,
                                                       resume_ok=resume_ok)
    if self.long_control_state == LongCtrlState.off:
      self.reset()
      output_accel = 0.

    elif self.long_control_state == LongCtrlState.stopping:
      output_accel = self.last_output_accel
      if output_accel > self.CP.stopAccel:
        output_accel = min(output_accel, 0.0)
        # GS 450h stationary-SET patch (2026-07-07): when already at standstill there is
        # nothing to stop smoothly - ramp fast to hold force so creep does not move the car
        # (0.2 took 10s to reach stopAccel=-2.0; 5.0 takes 0.4s)
        # Moving: upstream's flat 1.0 (2026-09-04, was CP.deprecated.stoppingDecelRate = 0.2).
        # Measured over 2.3h / 41 stops: entry accel is p50 -0.30, so this costs 0.34s / 5cm at
        # the median, but it cuts the shallow tail (a0 -0.15 took 1.75s / 48cm to stop) which is
        # the dawdle the driver feels. Numbers from _e2e_accel_ceiling.py --ramp.
        decel_rate = 5.0 if CS.standstill else 1.0
        output_accel -= decel_rate * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
