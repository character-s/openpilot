"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.modeld_v2.blendful import Blendful
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

X = np.array(ModelConstants.X_IDXS)
BUILD, RELEASE = 0.5, 2.0


def frame(center=0.0, plan=0.0, k=0.0, probs=(0.9, 0.9), stds=(0.1, 0.1), width=3.5, slope=0.0):
  road = k * X ** 2 / 2 + slope * X  # slope > 0: 線が先で左へ寄る = 車が車線に対して右を向いている
  lines = np.zeros((1, 4, 33, 2))
  lane_stds = np.ones((1, 4, 33, 2))
  for i, off in enumerate((-1.5 * width, -width / 2, width / 2, 1.5 * width)):
    lines[0, i, :, 0] = road + center + off
  lane_stds[0, 1, :, 0], lane_stds[0, 2, :, 0] = stds
  lane_probs = np.zeros((1, 8))
  lane_probs[0, 3], lane_probs[0, 5] = probs
  path = np.zeros((1, 33, 3))
  path[0, :, 0] = X
  path[0, :, 1] = road + plan * np.clip(X / 10, 0, 1)
  return {'lane_lines': lines, 'lane_lines_stds': lane_stds, 'lane_lines_prob': lane_probs, 'plan': path}


def drive(bf, seconds, v=20.0, k=0.0, **kw):
  out = [bf.update(frame(k=k, **kw), v, True, k) for _ in range(int(seconds / DT_MDL))]
  return np.array(out) * v ** 2  # m/s^2


class TestBlendful(OpenpilotTestCase):
  def setup_method(self):
    self.bf = Blendful()

  def test_centered_adds_nothing(self):
    push = drive(self.bf, 5.0)
    assert self.bf.trust > 0.9
    assert np.abs(push).max() < 1e-3

  def test_pulls_toward_lane_center(self):
    push = drive(self.bf, 5.0, center=0.5)
    assert push[-1] > 0.02  # + y side, where the lane center is
    assert self.bf.trust > 0.9
    assert np.all(np.diff(push) <= BUILD * DT_MDL + 1e-9)

  def test_flipping_sides_builds_slowly(self):
    k = 0.002  # 0.8 m/s^2 at 20 m/s
    drive(self.bf, 5.0, k=k, center=0.5)
    push = drive(self.bf, 5.0, k=k, center=-0.5)
    rate = np.diff(push) / DT_MDL
    assert push[-1] < 0
    assert np.all(rate >= -RELEASE - 1e-6)
    assert np.all(rate[push[:-1] <= 0] >= -BUILD - 1e-6)  # once past zero it builds
    crossing = push[1:][(push[:-1] > 0) & (push[1:] <= 0)]
    assert len(crossing) and np.all(crossing >= -BUILD * DT_MDL - 1e-9)

  def test_weak_lines_drive_laneless(self):
    clear = drive(Blendful(), 5.0, center=0.5)[-1]
    faint = Blendful()
    push = drive(faint, 5.0, center=0.5, probs=(0.1, 0.1))
    assert faint.trust < 0.25 and abs(push[-1]) < 0.25 * clear  # faint lines = trust and push scale down
    blurry = Blendful()
    push = drive(blurry, 5.0, center=0.5, stds=(0.8, 0.8))
    assert blurry.trust == 0.0 and np.abs(push).max() < 1e-6  # lines too uncertain = laneless

  def test_lone_line_holds_through_a_blink_in_a_curve(self):
    k, v = 1 / 20, 6.0
    drive(self.bf, 6.0, v=v, k=k, plan=0.4, probs=(0.9, 0.05))
    before = self.bf.trust
    drive(self.bf, 0.25, v=v, k=k, plan=0.4, probs=(0.1, 0.05), stds=(1.0, 0.1))
    bottom = self.bf.trust
    drive(self.bf, 2.0, v=v, k=k, plan=0.4, probs=(0.9, 0.05))
    assert before > 0.5
    assert bottom > 0.5 * before  # keeps latched through flicker
    assert self.bf.trust > 0.85 * before

  def test_yields_when_the_model_goes_around(self):
    drive(self.bf, 6.0)
    push = drive(self.bf, 3.0, plan=0.8)
    assert self.bf.going_around
    assert self.bf.trust < 0.05
    assert abs(push[-1]) < 0.01

  def test_fit_does_not_grow_while_trust_falls(self):
    drive(self.bf, 5.0, center=0.3)
    for i in range(40):
      dk, trust = self.bf.dk, self.bf.trust
      p = 0.9 - 0.02 * i
      self.bf.update(frame(center=0.3, plan=-0.02 * i, probs=(p, p)), 20.0, True, 0.0)
      if self.bf.trust < trust:
        assert abs(self.bf.dk) <= abs(dk) + 1e-12

  def test_toggle_off_returns_zero_and_resets(self):
    drive(self.bf, 5.0, center=0.5)
    assert self.bf.curvature != 0.0
    assert self.bf.update(frame(center=0.5), 20.0, True, 0.0, enabled=False) == 0.0
    assert self.bf.curvature == 0.0 and self.bf.trust == 0.0

  def test_wide_lines_hand_back_to_model(self):
    """GS450h: 線の間隔が広い道 (センターラインの無い対面通行など) では真ん中へ引き戻さず、モデルの左寄りに任せる。"""
    normal = drive(Blendful(), 8.0, center=0.6, width=3.5)[-1]
    push = drive(self.bf, 8.0, center=0.6, width=4.6)  # 車は 2 本の真ん中より 0.6m 左
    assert self.bf.trust < 1e-6 and abs(push[-1]) < 1e-4
    # 線が立ち上がる最初の数フレーム (両側の確からしさ 0.2 未満 = 幅を見る前) の引きは普通の車線の数 % に留まる
    assert np.abs(push).max() < 0.05 * normal

  def test_wide_band_fades_partially(self):
    full = Blendful()
    drive(full, 8.0, center=0.4, width=3.8)
    half = Blendful()
    drive(half, 8.0, center=0.4, width=4.2)
    assert full.trust > 0.9
    assert 0.3 < half.trust < 0.7  # 4.0-4.4m の間で段階的に落とす

  def test_highway_wide_lane_still_centers(self):
    """3.75m 車線の幹線は 3.8-4.0m に出る (11-02 実測、77km/h)。そこは今までどおり中心へ寄せる。"""
    push = drive(self.bf, 5.0, center=0.5, width=3.95)
    assert self.bf.trust > 0.9 and push[-1] > 0.02

  def test_wide_width_is_not_learned(self):
    """広い道の幅を「普段の車線幅」として覚えると、片側の線だけのときに目標が右へずれる。"""
    drive(self.bf, 30.0, width=3.2)
    learned = self.bf.lane_width.x
    drive(self.bf, 30.0, width=4.6)
    assert abs(self.bf.lane_width.x - learned) < 1e-6

  def test_heading_turns_back_toward_the_lane(self):
    """GS450h: 高速で車線に対して斜めを向いていたら、向きを戻す側へすぐ足す (中心寄せのなまし 0.5 秒を通さない)。"""
    v = 29.0
    drive(self.bf, 5.0, v=v)
    push = drive(self.bf, 0.5, v=v, slope=0.003)  # 右へ 0.17 度
    assert push[-1] > 0.03  # 左 (+) へ戻す
    assert self.bf.heading > 0
    push = drive(self.bf, 1.0, v=v, slope=-0.003)
    assert push[-1] < -0.03

  def test_heading_is_only_for_highway_straights(self):
    for kw in ({'v': 15.0}, {'v': 21.5}, {'v': 29.0, 'k': 0.0008}):  # 54km/h / 77km/h / 105km/h で 0.67 m/s^2 のカーブ
      bf = Blendful()
      drive(bf, 5.0, **kw)
      drive(bf, 1.0, slope=0.003, **kw)
      assert bf.heading == 0.0, kw

  def test_heading_lets_go_with_trust(self):
    v = 29.0
    drive(self.bf, 5.0, v=v, slope=0.003)
    assert self.bf.heading > 0
    self.bf.update(frame(slope=0.003), v, False, 0.0)  # 車線変更中
    assert self.bf.heading == 0.0
    wide = Blendful()
    drive(wide, 5.0, v=v, slope=0.003, width=4.6)
    assert abs(wide.heading) < 1e-9

  def test_highway_centering_ignores_a_wobbling_view(self):
    """GS450h: 高速の直線では、線の見え方の 4 秒周期の揺らぎ (位置 ±10cm) を寄せ量に素通ししない。一定の片寄りはそのまま寄せる。"""
    v = 29.0
    def wobble(bf, v):
      t = np.arange(int(20 / DT_MDL)) * DT_MDL
      out = [bf.update(frame(center=0.3 + 0.1 * np.sin(2 * np.pi * 0.25 * s)), v, True, 0.0) for s in t]
      return np.array(out[200:]) * v ** 2
    hw = wobble(self.bf, v)
    town = wobble(Blendful(), 18.0)  # 65km/h は今までどおり
    assert hw.mean() > 0.02  # 0.3m の片寄りは寄せる
    assert hw.std() < 0.5 * town.std()  # 揺らぎへの追従は半分以下 (寄せの横加速度は先読み距離で割るので速度によらない)
    assert self.bf.dk_slow != self.bf.dk

  def test_highway_yields_as_fast_as_before(self):
    v = 29.0
    drive(self.bf, 8.0, v=v)
    push = drive(self.bf, 3.0, v=v, plan=0.8)
    assert self.bf.going_around and self.bf.trust < 0.05
    assert abs(push[-1]) < 0.01

  def test_fades_below_min_speed(self):
    drive(self.bf, 5.0, v=10.0, center=0.5)
    start = self.bf.curvature
    out = np.array([self.bf.update(frame(center=0.5), 4.0, True, 0.0) for _ in range(100)])
    assert self.bf.trust == 0.0
    assert abs(out[0]) < abs(start)
    assert abs(out[-1]) < 1e-6
