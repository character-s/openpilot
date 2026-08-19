import numpy as np
from cereal import car
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N
from openpilot.common.pid import PIDController
from openpilot.selfdrive.modeld.constants import ModelConstants

CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]

LongCtrlState = car.CarControl.Actuators.LongControlState

# PLN-3 (2026-08-20): resume hysteresis on the stopping->pid transition.
# GS 450h has no second gate against spurious resumes: PCM standstill hold
# (cruiseState.standstill) never engages here — hold is only the sustained
# stopAccel command from the stopping state. shouldStop is computed from
# *planned* velocity, so a night model that fails to see a red light flips
# it open for a few frames and the car creeps into the crosswalk
# (route 142, 2026-08-19). Require GO to persist before releasing the hold:
#   - with a sustained lead: short hysteresis (lead pulling away is real evidence)
#   - with no lead (stopped at the head of a queue): no automatic resume at all;
#     the driver resumes with gas/RES. Finite values here would only delay the
#     creep, since a blind model keeps emitting GO indefinitely.
# The v_ego escape below keeps a manual (gas) launch from fighting the brake.
PLN3_GO_SUSTAIN_LEAD = 0.5    # [s] shouldStop must stay false this long (lead present)
PLN3_GO_SUSTAIN_NOLEAD = None  # [s] same with no lead; None disables auto resume
PLN3_LEAD_SUSTAIN = 1.0       # [s] hasLead must persist this long to count (ghost leads)


def long_control_state_trans(CP, CP_SP, active, long_control_state, v_ego,
                             should_stop, brake_pressed, cruise_standstill,
                             resume_ok=True):
  # Gas Interceptor
  cruise_standstill = cruise_standstill and not CP_SP.enableGasInterceptor

  stopping_condition = should_stop
  starting_condition = (not should_stop and
                        not cruise_standstill and
                        not brake_pressed)
  started_condition = v_ego > CP.vEgoStarting

  if not active:
    long_control_state = LongCtrlState.off

  else:
    if long_control_state == LongCtrlState.off:
      if not starting_condition:
        long_control_state = LongCtrlState.stopping
      else:
        if starting_condition and CP.startingState:
          long_control_state = LongCtrlState.starting
        else:
          long_control_state = LongCtrlState.pid

    elif long_control_state == LongCtrlState.stopping:
      if starting_condition and resume_ok and CP.startingState:
        long_control_state = LongCtrlState.starting
      elif starting_condition and resume_ok:
        long_control_state = LongCtrlState.pid

    elif long_control_state in [LongCtrlState.starting, LongCtrlState.pid]:
      if stopping_condition:
        long_control_state = LongCtrlState.stopping
      elif started_condition:
        long_control_state = LongCtrlState.pid
  return long_control_state

class LongControl:
  def __init__(self, CP, CP_SP):
    self.CP = CP
    self.CP_SP = CP_SP
    self.long_control_state = LongCtrlState.off
    self.pid = PIDController((CP.longitudinalTuning.kpBP, CP.longitudinalTuning.kpV),
                             (CP.longitudinalTuning.kiBP, CP.longitudinalTuning.kiV),
                             rate=1 / DT_CTRL)
    self.last_output_accel = 0.0
    # PLN-3 resume hysteresis timers
    self.go_timer = 0.0    # seconds shouldStop has been continuously false
    self.lead_timer = 0.0  # seconds hasLead has been continuously true

  def reset(self):
    self.pid.reset()

  def update(self, active, CS, a_target, should_stop, accel_limits, has_lead=False):
    """Update longitudinal control. This updates the state machine and runs a PID loop"""
    self.pid.neg_limit = accel_limits[0]
    self.pid.pos_limit = accel_limits[1]

    # PLN-3: track GO persistence and lead persistence at DT_CTRL rate
    self.go_timer = 0.0 if should_stop else min(self.go_timer + DT_CTRL, 10.0)
    self.lead_timer = min(self.lead_timer + DT_CTRL, 10.0) if has_lead else 0.0

    go_sustain = PLN3_GO_SUSTAIN_LEAD if self.lead_timer >= PLN3_LEAD_SUSTAIN else PLN3_GO_SUSTAIN_NOLEAD
    resume_ok = go_sustain is not None and self.go_timer >= go_sustain
    # escape: the car is already moving (driver launched with gas) - release the
    # hold so the stopping state does not drag the brake against them
    if CS.vEgo > self.CP.vEgoStarting and not should_stop:
      resume_ok = True

    self.long_control_state = long_control_state_trans(self.CP, self.CP_SP, active, self.long_control_state, CS.vEgo,
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
        decel_rate = 5.0 if CS.standstill else self.CP.stoppingDecelRate
        output_accel -= decel_rate * DT_CTRL
      self.reset()

    elif self.long_control_state == LongCtrlState.starting:
      output_accel = self.CP.startAccel
      self.reset()

    else:  # LongCtrlState.pid
      error = a_target - CS.aEgo
      output_accel = self.pid.update(error, speed=CS.vEgo,
                                     feedforward=a_target)

    self.last_output_accel = np.clip(output_accel, accel_limits[0], accel_limits[1])
    return self.last_output_accel
