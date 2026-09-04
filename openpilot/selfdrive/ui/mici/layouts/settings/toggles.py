from collections.abc import Callable

from openpilot.cereal import log

from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigMultiParamToggle, BigToggle, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.lane_centering import LaneCenteringChoice, LaneCenteringToggle
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationCircleButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering_params as lcp

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants


class ExperimentalModeConfirmPage(NavScroller):
  def __init__(self, on_confirm: Callable[[], None]):
    super().__init__()

    accept = BigConfirmationCircleButton("enable\nexperimental mode",
                                         gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 64, 64),
                                         lambda: self.dismiss(on_confirm))

    self._scroller.add_widgets([
      GreyBigButton("enabling\nexperimental mode", "scroll to continue",
                    gui_app.texture("icons_mici/setup/warning.png", 64, 64)),
      GreyBigButton("", "openpilot defaults to driving in chill mode."),
      GreyBigButton("", "Experimental mode enables alpha-level features that aren't ready for chill mode."),
      GreyBigButton("End-to-End Longitudinal Control"),
      GreyBigButton("", "Let the driving model control the gas and brakes."),
      GreyBigButton("", "openpilot will drive as it thinks a human would, including stopping for red lights and stop signs."),
      GreyBigButton("", "The set speed will only act as an upper bound."),
      GreyBigButton("", "This is an alpha quality feature; mistakes should be expected."),
      GreyBigButton("New Driving Visualization"),
      GreyBigButton("", "The path will change colors to communicate acceleration intent."),
      GreyBigButton("", "Red for braking, green for acceleration, and gray for coasting."),
      accept,
    ])


class TogglesLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._personality_toggle = BigMultiParamToggle("driving personality", "LongitudinalPersonality", ["aggressive", "standard", "relaxed"])
    self._experimental_btn = BigToggle("experimental mode", initial_state=ui_state.params.get_bool("ExperimentalMode"),
                                       toggle_callback=self._on_experimental_mode)
    # DEC (experimental mode の中で ACC と e2e のどちらを使うかをモデルに選ばせる設定) は旧 UI にしか無く、
    # c4 では切り替える手段が無かったのでここに出す。exp との連動は下の set_enabled を参照。
    self._dec_toggle = BigParamControl("dynamic experimental control", "DynamicExperimentalControl")
    # Lane Centering (StarPilot 由来の幾何補正)。⚠ この 4 つは openpilot の Params ではなく
    # `/data/params_fork/d` に保存している — params に置くと `clearAll` のホワイトリストから外れて
    # manager 起動のたびに消えるため。読み書きは lane_centering_params が持つ。
    # ⚠ 横制御なので DEC/exp の縦制御ゲートには連動させない。
    self._lane_centering_toggle = LaneCenteringToggle("lane centering", lcp.KEY_ENABLED)
    # 詳細 3 つは親が ON のときだけ出す。⚠ callable を渡すのは、親を押した瞬間に出したいため
    # (_update_toggles は show_event と engaged 遷移でしか回らない)。
    self._lane_centering_details = (
      LaneCenteringChoice("center offset", lcp.KEY_OFFSET, lcp.OFFSET_CHOICES, lcp.offset_label),
      LaneCenteringChoice("yield to model", lcp.KEY_AUTHORITY, lcp.AUTHORITY_CHOICES, lcp.authority_label),
      LaneCenteringToggle("pause on signal", lcp.KEY_PAUSE_ON_SIGNAL),
    )
    for item in self._lane_centering_details:
      item.set_visible(lambda: self._lane_centering_toggle._checked)
    is_metric_toggle = BigParamControl("use metric units", "IsMetric")
    ldw_toggle = BigParamControl("lane departure warnings", "IsLdwEnabled")
    always_on_dm_toggle = BigParamControl("always-on driver monitor", "AlwaysOnDM")
    record_front = BigParamControl("record & upload cabin camera", "RecordFront", toggle_callback=restart_needed_callback)
    record_mic = BigParamControl("record & upload mic audio", "RecordAudio", toggle_callback=restart_needed_callback)
    enable_openpilot = BigParamControl("enable sunnypilot", "OpenpilotEnabledToggle", toggle_callback=restart_needed_callback)

    self._scroller.add_widgets([
      self._personality_toggle,
      self._experimental_btn,
      self._dec_toggle,
      self._lane_centering_toggle,
      *self._lane_centering_details,
      is_metric_toggle,
      ldw_toggle,
      always_on_dm_toggle,
      record_front,
      record_mic,
      enable_openpilot,
    ])

    # Toggle lists
    self._refresh_toggles = (
      ("ExperimentalMode", self._experimental_btn),
      ("DynamicExperimentalControl", self._dec_toggle),
      ("IsMetric", is_metric_toggle),
      ("IsLdwEnabled", ldw_toggle),
      ("AlwaysOnDM", always_on_dm_toggle),
      ("RecordFront", record_front),
      ("RecordAudio", record_mic),
      ("OpenpilotEnabledToggle", enable_openpilot),
    )

    # exp OFF では planner が DEC を見ない (longitudinal_planner.py) ので、ON にできないよう灰色にする。
    # param でなくトグルの表示状態を見るのは、毎フレーム param を読まずに済み、かつ確認ダイアログ
    # 未確定の間も見た目と一致するため。
    self._dec_toggle.set_enabled(lambda: self._experimental_btn._checked)
    enable_openpilot.set_enabled(lambda: not ui_state.engaged)
    record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    record_mic.set_enabled(lambda: not ui_state.engaged)

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    super()._update_state()

    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._personality_toggle.set_value(self._personality_toggle._options[personality])
      ui_state.personality = personality

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    # CP gating for experimental mode
    if ui_state.CP is not None:
      if ui_state.has_longitudinal_control:
        self._experimental_btn.set_visible(True)
        self._personality_toggle.set_visible(True)
        self._dec_toggle.set_visible(True)
      else:
        # no long for now
        self._experimental_btn.set_visible(False)
        self._experimental_btn.set_checked(False)
        self._personality_toggle.set_visible(False)
        self._dec_toggle.set_visible(False)
        ui_state.params.remove("ExperimentalMode")

    # Refresh toggles from params to mirror external changes
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))

    # Lane Centering は Params ではなく `/data/params_fork/d` に保存しているので自前で取り込む。
    # SSH で直接書かれた場合もここで画面に反映される。
    self._lane_centering_toggle.refresh()
    for item in self._lane_centering_details:
      item.refresh()

  def _on_experimental_mode(self, state: bool):
    if state and not ui_state.params.get_bool("ExperimentalModeConfirmed"):
      # Don't show enabled state until confirm
      self._experimental_btn.set_checked(False)

      def on_confirm():
        ui_state.params.put_bool("ExperimentalModeConfirmed", True)
        ui_state.params.put_bool("ExperimentalMode", True)
        self._experimental_btn.set_checked(True)

      gui_app.push_widget(ExperimentalModeConfirmPage(on_confirm))
    else:
      ui_state.params.put_bool("ExperimentalMode", state)
