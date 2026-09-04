"""
LaneCenteringController のテスト。

StarPilot (MIT) の selfdrive/controls/tests/test_lane_centering.py を移植 + GS 向けに追加
(実効ゲインの検算 / 上限クリップ / 帯と幅門が原版のままであること / 幅急変ガード / params 経路)。
"""
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering_params as lcp
from openpilot.sunnypilot.selfdrive.controls.lib.lane_centering import (
  LaneCenteringController, _deadband_for, _gain_for, _lookahead_min_for,
)


_V_EGO = 20.0
_XS = np.linspace(0.0, 50.0, 52)


def _path(y, y_std=0.1):
  return SimpleNamespace(
    x=_XS.copy(),
    y=np.full_like(_XS, float(y)),
    yStd=np.full_like(_XS, float(y_std)),
  )


def _model(left=-1.8, right=1.8, model_y=0.0, lane_prob=0.9, lane_std=0.1, path_std=0.1, lane_change=0):
  return SimpleNamespace(
    laneLines=[_path(0.0), _path(left), _path(right), _path(0.0)],
    laneLineProbs=[0.0, lane_prob, lane_prob, 0.0],
    laneLineStds=[0.0, lane_std, lane_std, 0.0],
    position=_path(model_y, path_std),
    meta=SimpleNamespace(laneChangeState=lane_change),
  )


def _update(controller, model, *, offset=0.0, authority=0.0, enabled=True, active=True, valid=True, speed=_V_EGO,
            pause_on_signal=False, turn_signal_active=False):
  controller.enabled = enabled
  controller.offset = offset
  controller.e2e_authority = authority
  controller.pause_on_signal = pause_on_signal
  lane_change = model.meta.laneChangeState != 0
  return controller.update(0.0, model, speed, active, valid, lane_change, turn_signal_active)


def _feed(controller, model, n, **kw):
  out = 0.0
  for _ in range(n):
    out = _update(controller, model, **kw)
  return out


def _converge(model, **kw):
  controller = LaneCenteringController()
  return controller, _feed(controller, model, 300, **kw)


def _narrow(**kw):
  return _model(left=-1.5, right=1.5, **kw)                # 幅 3.0m、誤差 0


def _widened():
  return _model(left=-1.5, right=2.5, path_std=0.6)        # 幅 4.0m、誤差 0.5m (path_std 大 = authority を効かせない)


def _settled_on_narrow():
  """幅 3.0m を authority=1.0 で 200 frame 慣らした controller (幅急変ガードの前提)。毎回新規に組む。"""
  c = LaneCenteringController()
  _feed(c, _narrow(), 200, authority=1.0)
  return c


_HARD_GATES = [{"active": False}, {"valid": False}, {"speed": 2.4}]   # 速度門 2.5 m/s (9km/h) の直下
_HIGH_SPEEDS = [12.5, 15.0, 20.0, 30.0]                                  # 低速スケジュールの上端 (12.5 m/s = 45km/h) 以上


@pytest.mark.parametrize("kwargs", [{"enabled": False}, *_HARD_GATES])
def test_hard_gates_are_noop(kwargs):
  assert _update(LaneCenteringController(), _model(left=-1.5, right=2.1), **kwargs) == 0.0


def test_lane_change_is_noop():
  assert _update(LaneCenteringController(), _model(left=-1.5, right=2.1, lane_change=1)) == 0.0


def test_turn_signal_fades_correction():
  model = _model(left=-1.5, right=2.1)
  controller, centered = _converge(model)
  fading = _update(controller, model, pause_on_signal=True, turn_signal_active=True)
  assert 0.0 < fading < centered

  fading = _feed(controller, model, 300, pause_on_signal=True, turn_signal_active=True)
  assert abs(fading) < 1e-6


def test_disabling_fades_correction():
  """enabled OFF は 1 frame で 0 にせず 0.2s で抜くこと。

  UI トグル (08-26) を足したことで **走行中に切れる**ようになった。それまで enabled は
  起動時に決まる値でしかなく reset() で実害が無かったが、今は舵に効いている補正を
  ユーザー操作で消すことになるので、即断だと切った瞬間に舵が跳ねる。
  """
  model = _model(left=-1.5, right=2.1)
  controller, centered = _converge(model)
  assert centered > 0.0

  fading = _update(controller, model, enabled=False)
  assert 0.0 < fading < centered                  # 1 frame では消えない

  fading = _feed(controller, model, 300, enabled=False)
  assert fading == 0.0                            # 最後は完全に 0 (計算からも降りる)


@pytest.mark.parametrize("gate", _HARD_GATES)
def test_hard_gates_still_drop_correction_immediately(gate):
  """latActive off / 低速 / モデル停止は即断のまま (enabled OFF の平滑化との対比)。
  補正が舵に出ていない場面なので平滑して残す方が有害 (再 engage で古い補正から再開する)。"""
  model = _model(left=-1.5, right=2.1)
  controller, centered = _converge(model)
  assert centered > 0.0
  assert _update(controller, model, **gate) == 0.0
  assert controller._correction == 0.0


def test_turn_signal_pause_can_be_disabled():
  model = _model(left=-1.5, right=2.1)
  controller, output = _converge(model)          # 決定的なので 1 回で controller と出力の両方が取れる
  signaled = _update(controller, model, turn_signal_active=True)
  assert signaled == pytest.approx(output, abs=1e-7)


@pytest.mark.parametrize(
  "field,value",
  [
    ("prob", np.nan),
    ("prob", 1.1),
    ("std", np.nan),
    ("std", -0.1),
  ],
)
def test_invalid_lane_confidence_is_rejected(field, value):
  model = _model(left=-1.5, right=2.1)
  values = model.laneLineProbs if field == "prob" else model.laneLineStds
  values[1] = value
  assert _update(LaneCenteringController(), model) == 0.0


def test_input_must_cover_lookahead():
  model = _model(left=-1.5, right=2.1)
  model.laneLines[1].x = model.laneLines[1].x[:10]
  model.laneLines[1].y = model.laneLines[1].y[:10]
  assert _update(LaneCenteringController(), model) == 0.0


def test_lane_center_error_steers_toward_center():
  _, right = _converge(_model(left=-1.5, right=2.1))
  _, left = _converge(_model(left=-2.1, right=1.5))
  assert right > 0.0
  assert left < 0.0


def test_small_center_error_does_not_chatter():
  _, output = _converge(_model(left=-1.75, right=1.85))
  assert output == 0.0


def test_offset_direction():
  _, right = _converge(_model(), offset=0.2)
  _, left = _converge(_model(), offset=-0.2)
  assert right > 0.0
  assert left < 0.0


def test_offset_is_reduced_in_narrow_lane():
  narrow = _model(left=-1.3, right=1.3)
  _, at_safe_limit = _converge(narrow, offset=0.2)
  _, above_safe_limit = _converge(narrow, offset=0.3)
  assert np.isclose(at_safe_limit, above_safe_limit)


def test_confident_e2e_path_can_fully_break_in():
  model = _model(left=-1.0, right=2.6, model_y=0.0, path_std=0.1)
  _, lane_authority = _converge(model, authority=0.0)
  _, e2e_authority = _converge(model, authority=1.0)
  assert lane_authority > 0.0
  assert abs(e2e_authority) < 1e-9


def test_uncertain_e2e_path_does_not_break_in():
  model = _model(left=-1.0, right=2.6, model_y=0.0, path_std=0.6)
  _, output = _converge(model, authority=1.0)
  assert output > 0.0


def test_e2e_authority_blends_lane_correction():
  model = _model(left=-1.2, right=2.4, model_y=0.0, path_std=0.1)
  _, lane_only = _converge(model, authority=0.0)
  _, blended = _converge(model, authority=0.5)
  _, e2e = _converge(model, authority=1.0)
  assert lane_only > blended > e2e >= 0.0


def test_confidence_loss_drops_filtered_correction():
  controller, output = _converge(_model(left=-1.5, right=2.1))
  assert output > 0.0
  unsure = _model(left=-1.5, right=2.1, lane_prob=0.2)
  fading = _update(controller, unsure)
  assert 0.0 < fading < output

  fading = _feed(controller, unsure, 300)
  assert abs(fading) < 1e-6


def test_correction_is_smoothed_and_capped():
  controller = LaneCenteringController()
  model = _model(left=0.0, right=3.0, path_std=0.6)
  first = _update(controller, model)
  _, steady = _converge(model)
  assert 0.0 < first < steady
  assert np.isclose(steady, 0.004 * 0.30, atol=1e-6)


# ── ここから GS 向けに追加 ─────────────────────────────────────────────

@pytest.mark.parametrize("speed", _HIGH_SPEEDS)
def test_effective_gain_is_speed_invariant_in_lat_accel(speed):
  """la 換算のゲインが 12.5-35 m/s (下端 = 低速スケジュールの上端) で 0.6 × (|err| - deadband) になること。

  「big の復元勾配と同オーダー = P だけで平衡点が動く」という採否判断の根拠なので、実装から出る値を固定する。
  """
  err = 0.3          # 中心が +0.3m 右 = 車が 0.3m 左寄り
  _, steady = _converge(_model(left=-1.5, right=2.1), speed=speed)
  lat_accel = steady * speed ** 2
  assert np.isclose(lat_accel, 0.6 * (err - 0.08), rtol=1e-3)


@pytest.mark.parametrize("speed", [5.5, 8.0, 10.0])
def test_low_speed_saturates_at_the_curvature_cap(speed):
  """低速では raw が上限 0.004 に当たり、曲率が 0.0012 1/m で頭打ちになること。

  lookahead の下限が 8m 固定なので、低速ほど「同じ横誤差に要る曲率」が大きくなる。
  結果として狭路カーブ (エッジ接触が集中する帯) では上限張り付き = 定数トルクになる。
  それでも big の復元項 g·d0 ≈ 0.001 より大きいので、平衡点を動かす向きは変わらない。
  """
  _, steady = _converge(_model(left=-1.5, right=2.1), speed=speed)
  # 09-02: 低速スケジュール後は cap 0.004 × 速度ゲイン (5.5/8.0 m/s = 1.0 → 0.004、10 m/s = 0.689 → 0.00276)。
  #   上限 _MAX_RAW_CORRECTION 自体は据え置きなので「頭打ちになる」構造は変わらない。
  assert np.isclose(steady, 0.004 * _gain_for(speed), rtol=1e-2)   # 300 frame の平滑残差 (tau 0.4s) を許す


@pytest.mark.parametrize("v", _HIGH_SPEEDS)
def test_lowspeed_schedule_leaves_high_speed_untouched(v):
  """低速スケジュールは 12.5 m/s (45km/h) 以上を一切変えないこと (user「速度が出ているときは真ん中で良い」)。"""
  assert _gain_for(v) == pytest.approx(0.30)
  assert _deadband_for(v) == pytest.approx(0.08)
  assert _lookahead_min_for(v) == pytest.approx(8.0)


def test_lowspeed_schedule_changes_the_low_end():
  """対になる側: 8 m/s 以下ではゲイン 1.0 / 不感帯 0.04 / lookahead 下限 6m になっていること。"""
  assert _gain_for(8.0) == pytest.approx(1.0)
  assert _deadband_for(8.0) == pytest.approx(0.04)
  assert _lookahead_min_for(5.0) == pytest.approx(6.0)


def test_lowspeed_acts_on_small_error_but_high_speed_does_not():
  """6cm の左寄り: 低速 (5.5 m/s) では不感帯 0.04 を超えて補正が出る / 20 m/s では不感帯 0.08 の中で 0 のまま。

  旧定数では低速帯の寄りが不感帯に丸ごと入っていて LC が何もしていなかった。
  """
  model = _model(left=-1.74, right=1.86)   # 中心 +0.06 (右) = 車が 6cm 左
  _, low = _converge(model, speed=5.5)
  _, high = _converge(model, speed=20.0)
  assert low > 1e-4
  assert high == 0.0


def test_narrow_lane_gate_stays_at_upstream_value():
  """幅門の下限は原版どおり 2.6m (2.6m 未満は 1% 台で、下げれば upstream との差分が増えるだけ)。"""
  _, at_2_7m = _converge(_model(left=-1.1, right=1.6))    # 幅 2.7m → 通る
  _, at_2_5m = _converge(_model(left=-1.0, right=1.5))    # 幅 2.5m → 落ちる
  assert at_2_7m > 0.0
  assert at_2_5m == 0.0


def test_default_authority_keeps_full_gain_below_break_in_band():
  """既定 authority 1.0 のままでも、break_in 帯 (0.15-0.50m) の手前の誤差では減衰しないこと。"""
  controller = LaneCenteringController()
  assert controller.e2e_authority == 1.0   # 既定は原版どおり (0 にすると構造変化の保護が消える)

  model = _model(left=-1.72, right=1.88, path_std=0.1)   # 誤差 0.08-0.15m 帯 = 減衰前
  _, with_default = _converge(model, authority=1.0)
  _, no_authority = _converge(model, authority=0.0)
  assert np.isclose(with_default, no_authority, rtol=1e-6)


def test_break_in_band_is_upstream_default():
  """break_in 帯は原版のまま (0.15-0.50m)。補正が山型なので帯を広げても平衡点はほぼ動かない
  (綱引き 0.455×(0.40-d) = f(d) の解は原版 0.245m / 0.70-0.90 版 0.218m = 差 2.7cm)。"""
  _, small = _converge(_model(left=-1.65, right=1.95, path_std=0.1), authority=1.0)  # 誤差 0.15m
  _, large = _converge(_model(left=-1.3, right=2.3, path_std=0.1), authority=1.0)    # 誤差 0.50m
  assert small > 0.0
  assert abs(large) < 1e-9


def test_width_jump_suspends_correction():
  """GS 変更点 3: 車線幅が急に広がった瞬間は補正を抜くこと (右折レーンの出現など)。

  08-26 実験: ガード無しだと幅 3.0→4.0m で la +0.234 m/s² = 広がった側へ引き込まれた。
  path_std が大きい (モデルも迷っている) 場面では authority の break_in が効かないので、
  幅の跳びを独立に見る必要がある。
  """
  controller = _settled_on_narrow()
  assert abs(_feed(controller, _widened(), 50, authority=1.0)) < 1e-9   # 跳んだ直後 0.5s
  assert _feed(controller, _widened(), 400, authority=1.0) > 0.0        # 新しい幅に馴染めば復帰する


# ── 保存先 (`/data/params_fork/d`) の読み経路と幅ガードの状態管理 ──────────────

@pytest.fixture(autouse=True)
def param_dir(tmp_path, monkeypatch):
  """⚠ **全テストで** 保存先を tmp に向ける。実機 (c4) で pytest を回したときに本物の
  `/data/params_fork/d` を読み書きしてしまわないため。

  ⚠ 差し替えるのは lcp 側。lane_centering.py は定義を持たずモジュール属性で引いている。
  """
  d = tmp_path / "d"
  d.mkdir()
  monkeypatch.setattr(lcp, 'PARAM_DIR', str(d))
  return d


def _write_params(**over):
  """UI が設定画面から書いたのと同じ状態を作る (4 つとも書く)。

  ⚠ キーワードは保存名そのまま (LaneCentering / LaneCenterOffset / ...)。
  """
  values = {
    lcp.KEY_ENABLED: True,
    lcp.KEY_OFFSET: 0.0,
    lcp.KEY_AUTHORITY: 1.0,
    lcp.KEY_PAUSE_ON_SIGNAL: True,
  }
  values.update(over)
  for key, value in values.items():
    assert lcp.write(key, value), key


def test_params_are_read_on_construction_and_clipped():
  _write_params(LaneCenterOffset=0.9, LaneCenteringE2EAuthority=5.0)
  c = LaneCenteringController()
  assert c.enabled is True
  assert c.offset == 0.3          # _MAX_OFFSET まで clip
  assert c.e2e_authority == 1.0   # [0, 1] に clip
  assert c.pause_on_signal is True


def test_unreadable_key_falls_back_to_its_default_not_a_stale_value(param_dir):
  """読めないキーは「既定値」になること (前回値を引きずらない)。

  ⚠ 前回値を維持してしまうと、「LaneCentering のファイルだけ置いた」ときに他の 3 つが
  更新されず、項目ごとに世代の違う値が混ざる。
  """
  _write_params(LaneCenterOffset=0.2, LaneCenteringE2EAuthority=0.4)
  c = LaneCenteringController()
  assert (c.enabled, c.offset, c.e2e_authority) == (True, 0.2, 0.4)

  # authority のファイルだけ無くなる → 既定 1.0 に戻る (0.4 を引きずらない)
  lcp.write(lcp.KEY_OFFSET, -0.25)
  (param_dir / lcp.KEY_AUTHORITY).unlink()
  c.update_params()
  assert c.enabled is True
  assert c.offset == -0.25        # 読めた項目はちゃんと更新される
  assert c.e2e_authority == 1.0   # 読めなかった項目は既定値


def test_values_come_from_the_file(param_dir):
  """値の出所は lcp.PARAM_DIR のファイル 1 本 (Params を経由しない理由は lcp モジュール docstring)。"""
  (param_dir / "LaneCentering").write_bytes(b"1")
  (param_dir / "LaneCenterOffset").write_bytes(b"-0.12")

  c = LaneCenteringController()
  assert c.enabled is True          # ファイルから拾えている
  assert c.offset == -0.12
  assert c.e2e_authority == 1.0     # ファイルが無い項目は既定値
  assert c.pause_on_signal is True


def test_read_failure_keeps_the_previous_values(monkeypatch):
  """読みがどんな形で失敗しても前回値のまま走り続けること。

  ⚠ ここは controlsd (安全上クリティカル) の中なので、保存層のどんな失敗も制御ループを
  落としてはいけない。
  """
  _write_params(LaneCenterOffset=0.2)
  c = LaneCenteringController()

  def boom(key):
    raise RuntimeError('storage layer exploded')

  monkeypatch.setattr(lcp, 'read_float', boom)
  c.update_params()
  assert c.offset == 0.2            # 前回値のまま
  assert c.enabled is True


def test_params_nan_is_rejected(param_dir):
  _write_params(LaneCenterOffset=0.2)
  c = LaneCenteringController()
  (param_dir / lcp.KEY_OFFSET).write_bytes(b"nan")
  c.update_params()
  assert c.offset == 0.2


def test_width_reference_survives_low_speed_gap():
  """停車 (v<5) を挟んでも幅の基準が残り、その後の幅急変を捕まえられること。

  reset() が _width_ref も消していると、発進直後に基準が現在幅で初期化されて
  width_jump = 0 になり、低速の交差点まわり = ガードが最も要る区間で素通りする。
  """
  c = _settled_on_narrow()
  _update(c, _narrow(), speed=4.9, authority=1.0)     # 停車 → reset パスを通す
  assert abs(_feed(c, _widened(), 50, authority=1.0)) < 1e-9


def test_width_reference_survives_out_of_gate_width():
  """幅が門 (2.6-4.8m) を外れて戻ってきたときもガードが効くこと。

  門外で基準を捨てると、門内に復帰した瞬間に現在幅で初期化されて跳びを見逃す。
  """
  too_wide = _model(left=-1.5, right=3.6)               # 幅 5.1m = 門外
  c = _settled_on_narrow()
  _feed(c, too_wide, 30, authority=1.0)
  assert abs(_feed(c, _widened(), 30, authority=1.0)) < 1e-9   # 幅 4.0m = 門内に復帰


def test_lane_change_drops_width_reference():
  """車線変更のときだけは基準を捨てること (走る車線自体が変わるため)。"""
  c = _settled_on_narrow()
  _update(c, _narrow(lane_change=1), authority=1.0)
  assert _feed(c, _widened(), 300, authority=1.0) > 0.0   # 新しい幅を基準に補正が復帰する


def test_reenabling_drops_stale_width_reference():
  """OFF→ON では幅の基準を捨てること。

  OFF の間は _raw_correction を通らないので _width_ref が「切った場所の幅」で止まる。
  別の幅の道路で ON に戻すと、それを構造変化と誤認して幅急変ガードが誤発動し、
  ON にした直後 (= 効きを確かめたい場面) だけ補正が出ない。
  """
  _write_params()
  c = _settled_on_narrow()
  assert c._width_ref == pytest.approx(3.0, abs=0.05)

  lcp.write(lcp.KEY_ENABLED, False)
  c.update_params()
  assert c.enabled is False
  assert c._width_ref == pytest.approx(3.0, abs=0.05)   # OFF にしただけでは捨てない

  lcp.write(lcp.KEY_ENABLED, True)
  c.update_params()
  assert c._width_ref == 0.0                            # ON に戻すときに捨てる

  assert _feed(c, _widened(), 300, authority=1.0) > 0.0   # 新しい幅で素直に効く


def test_staying_enabled_keeps_width_reference():
  """ON のまま params を読み直しても基準は捨てないこと。

  update_params() は 1Hz で回るので、ここで毎回捨てると幅急変ガードが恒久的に死ぬ。
  """
  _write_params()
  c = _settled_on_narrow()
  ref = c._width_ref

  c.update_params()                                     # ON → ON
  assert c._width_ref == ref
  assert abs(_feed(c, _widened(), 50, authority=1.0)) < 1e-9   # ガードは生きている


# ── controlsd から呼ばれる入口 (apply) ──────────────────────────────────

class _SM:
  """controlsd の SubMaster のうち apply() が触るぶんだけ。"""
  def __init__(self, model, frame=0, checks=True):
    self._model = model
    self.frame = frame
    self._checks = checks

  def __getitem__(self, key):
    assert key == 'modelV2'
    return self._model

  def all_checks(self, services):
    return self._checks


def test_apply_routes_to_update_and_respects_maneuver():
  model = _model(left=-1.5, right=2.1)
  cs = SimpleNamespace(vEgo=_V_EGO, leftBlinker=False, rightBlinker=False)
  cc = SimpleNamespace(latActive=True)
  # ⚠ apply() は 1Hz で update_params() を呼ぶので、属性を直接セットしても上書きされる
  #   (保存先が読めない環境では既定値 = OFF になる = 安全側の正しい挙動)。ファイルに書いて渡す。
  _write_params()
  c = LaneCenteringController()
  assert c.enabled is True

  out = 0.0
  for i in range(300):
    out = c.apply(0.0, _SM(model, frame=i), cs, cc, False, False)
  assert out > 0.0

  # maneuver 中は素通し + 状態も持ち越さない
  assert c.apply(0.0, _SM(model, frame=301), cs, cc, True, False) == 0.0
  assert c._correction == 0.0


def test_apply_passes_through_gates():
  model = _model(left=-1.5, right=2.1)
  cs = SimpleNamespace(vEgo=_V_EGO, leftBlinker=True, rightBlinker=False)
  cc = SimpleNamespace(latActive=True)
  _write_params(LaneCenteringPauseOnSignal=True)
  c = LaneCenteringController()
  # ウインカー中 (CS から拾う) / modelV2 が invalid / lane_change の 3 経路
  assert c.apply(0.0, _SM(model, frame=1), cs, cc, False, False) == 0.0
  cs2 = SimpleNamespace(vEgo=_V_EGO, leftBlinker=False, rightBlinker=False)
  assert c.apply(0.0, _SM(model, frame=1, checks=False), cs2, cc, False, False) == 0.0
  assert c.apply(0.0, _SM(model, frame=1), cs2, cc, False, True) == 0.0
