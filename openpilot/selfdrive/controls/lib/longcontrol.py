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

# Lead-less auto resume. This is what makes the car pull away from a red light on its own,
# so it is gated on which model is actually driving:
#
#   16D (small)  PLN3_GO_SUSTAIN_NOLEAD = None  -> disabled.
#                Measured false GO 3/9 while stopped. That is why this was turned off.
#   TT  (big)    PLN3_GO_SUSTAIN_NOLEAD_BIG     -> 1.5s.
#                08-24 first drive: all 14 signal stops needed the driver's gas pedal, and the
#                only thing holding the car was resume_ok - the model's GO led the human's
#                launch by p50 1.2s and the plan had already flipped.
#                Started at 2.0s; the driver called it too slow on the first drive where it
#                actually fired (2026-08-25), so 1.5s. Route 162 (110 seg, same day) backs
#                this: all 5 lead-less auto launches sat at go_timer 2.01-2.02s = the sustain
#                itself was the bottleneck, and every aborted GO burst was <= 1.35s, so 1.5s
#                would have launched exactly the same 5 and no flyer. ⚠ that longest flyer
#                (1.35s) leaves only 0.15s of margin - if a false GO ever launches the car,
#                this is the number to put back up, not something to tune further down.
#
# ⚠ modeld falls back to the small model whenever the eGPU is missing or fails to load, so this
#   MUST follow the model at runtime, not a build-time constant.
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
        # (stock stoppingDecelRate=0.8 takes 2.5s to reach stopAccel=-2.0; 5.0 takes 0.4s)
        # NOTE: upstream switched to a flat 1.0 here; we keep CP.stoppingDecelRate (GS = 0.2/0.3)
        decel_rate = 5.0 if CS.standstill else self.CP.deprecated.stoppingDecelRate
        output_accel -= decel_rate * DT_CTRL  # m/s^2/s while trying to stop
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
