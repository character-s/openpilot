from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

class MockLeadOne:
  # vLead / dRel are read by the launch-fix v3 branch (_lead_departing); without
  # them the mock raises AttributeError as soon as a lead is present.
  def __init__(self, present=0.0, vLead=0.0, dRel=100.0):
    self.present = present
    self.vLead = vLead
    self.dRel = dRel

class MockRadarState:
  def __init__(self, present=0.0):
    self.leadOne = MockLeadOne(present=present)

class MockCarState:
  def __init__(self, vEgo=0.0, vCruise=0.0, standstill=False):
    self.vEgo = vEgo
    self.vCruise = vCruise
    self.standstill = standstill

class MockModelData:
  def __init__(self, valid=True):
    size = 33 if valid else 10  # incomplete if invalid
    self.position = type("Pos", (), {"x": [0.0] * size})()
    self.orientation = type("Ori", (), {"x": [0.0] * size})()

class MockSelfDriveState:
  def __init__(self, experimentalMode=False):
    self.experimentalMode = experimentalMode

class MockParams:
  def get_bool(self, name):
    return True

def default_sm():
  sm = {
    'carState': MockCarState(vEgo=10.0, vCruise=20.0),
    'radarState': MockRadarState(present=1.0),
    'modelV2': MockModelData(valid=True),
    'selfdriveState': MockSelfDriveState(experimentalMode=True),
  }
  return sm

def mock_cp():
  class CP:
    radarUnavailable = False
  return CP()

def mock_mpc():
  class MPC:
    crash_cnt = 0
  return MPC()

# Fake Kalman Filter that always returns a given value
class FakeKalman:
  def __init__(self, value=1.0):
    self.value = value
  def add_data(self, v): pass
  def get_value(self): return self.value
  def get_confidence(self): return 1.0
  def reset_data(self): pass

class TestDynamicExperimentalController(OpenpilotTestCase):
  def test_initial_mode_is_acc(self, mock_cp, mock_mpc):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    assert controller.mode() == "acc"

  def test_standstill_triggers_blended(self, mock_cp, mock_mpc, default_sm):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    default_sm['carState'].standstill = True
    for _ in range(10):
      controller.update(default_sm)
    assert controller.mode() == "blended"

  def test_emergency_blended_on_fcw(self, mock_cp, mock_mpc, default_sm):
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    mock_mpc.crash_cnt = 1  # simulate FCW
    for _ in range(2):
      controller.update(default_sm)
    assert controller.mode() == "blended"

  def test_lead_with_slowdown_prefers_blended(self, mock_cp, mock_mpc, default_sm):
    """GS 450h reorder: a lead is present AND we need to slow down -> blended (e2e).

    With the upstream order "lead detected -> acc" returns first, so the moment
    radar acquires a target the planner drops e2e and loses the model's visual
    look-ahead braking. Measured on route 015 (51 lead-decel scenes): upstream
    order picked blended 0.2% of frames, this order 65.4%.
    """
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    controller._lead_filter = FakeKalman(value=1.0)       # ty: ignore[invalid-assignment]
    controller._slow_down_filter = FakeKalman(value=1.0)  # ty: ignore[invalid-assignment]

    for _ in range(10):
      controller.update(default_sm)

    assert controller.mode() == "blended"

  def test_lead_without_slowdown_stays_acc(self, mock_cp, mock_mpc, default_sm):
    """Steady following (no slow down needed) must stay on acc.

    This is what DEC is for: cruising on MPC so the set speed is actually
    reached, instead of the model settling below it.
    """
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    controller._lead_filter = FakeKalman(value=1.0)       # ty: ignore[invalid-assignment]
    controller._slow_down_filter = FakeKalman(value=0.0)  # ty: ignore[invalid-assignment]

    for _ in range(10):
      controller.update(default_sm)

    assert controller.mode() == "acc"

  def test_departing_lead_beats_slowdown(self, mock_cp, mock_mpc, default_sm):
    """launch fix v3 must stay ahead of the slow_down branch after the reorder.

    If slow_down claimed the frame first, the departure stall would come back
    (cf/seg77: blended held for ~4s while the lead kept pulling away).
    """
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())
    controller._slow_down_filter = FakeKalman(value=1.0)  # ty: ignore[invalid-assignment]
    default_sm['carState'].standstill = True
    default_sm['radarState'].leadOne.vLead = 1.0
    default_sm['radarState'].leadOne.dRel = 10.0

    # The first 3 frames still have standstill_count <= 3, so they fall through to
    # slow_down and urgency=1.0 pins an emergency blended (timeout 15 / override 20).
    # Run long enough for launch fix to take the mode back — which mirrors the real
    # case, where the lead starts moving a second or so after we stop.
    for _ in range(40):
      controller.update(default_sm)

    assert controller.mode() == "acc"

  def test_radarless_slowdown_triggers_blended(self, mock_cp, mock_mpc, default_sm):
    mock_cp.radarUnavailable = True
    controller = DynamicExperimentalController(mock_cp, mock_mpc, params=MockParams())

    # Force conditions to simulate slowdown
    controller._slow_down_filter = FakeKalman(value=1.0)  # ty: ignore[invalid-assignment]
    controller._v_ego_kph = 35.0
    default_sm['modelV2'] = MockModelData(valid=False)  # Incomplete trajectory

    for _ in range(3):
      controller.update(default_sm)

    assert controller.mode() == "blended"
