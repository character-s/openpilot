"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.modeld_v2.constants import ModelConstants

LANE_WIDTH_DEFAULT = 3.6  # m
LANE_WIDTH_CLIP = (2.8, 4.5)  # m
# GS450h: 線の間隔が普通の車線より広いとき (センターラインの無い対面通行で、両側の外側線を自分の車線と見ている等) は
# 2 本の真ん中 = 道路の真ん中へ寄せてしまうので、効きを落としてモデル本来の左寄りに任せる (LC の幅門 2.6-4.8m と同じ考え方)。
# 11-02 実測 (横制御中・両側の線 >0.5): Blendful OFF の車の位置は 3.8m までは真ん中 ±0.1m、4.0m を超えると 0.3-0.9m 左。
# ON はそこを約 0.25m 真ん中へ引き戻していた。⚠ 3.75m 車線の幹線 (77km/h) は 3.8-4.0m に出るので 4.0m までは触らない。
WIDE_LANE_FADE = (4.0, 4.4)  # m: この間で効きを 1→0 に落とす
# GS450h: 高速で 4-5 秒周期の緩い蛇行が出る (11-02 新東名 105km/h、OFF 比で揺れ +30%)。中心のずれだけを見て寄せると、
# 線の見え方のゆっくりした揺らぎ (位置) を追いかけて車を振る。線の傾き = 車線に対する車の向きに逆らう項を足して揺れを止める。
# シミュレーション (10-31 の rlog から同定した車とモデル + 本物の Blendful) では揺れ p50/p90 3.6/5.3 → 2.3/3.6cm、中心寄せは落ちない。
HEADING_DAMP = 0.6  # 1/s: 向きの戻り速さ (曲率 = HEADING_DAMP / v * 傾き。105km/h で 0.02 1/m/rad)
HEADING_SPEED = (22.2, 27.8)  # m/s (80-100 km/h): 確かめたのは高速だけなので、それより下では足さない (寄せ量のなましも同じ帯)
HEADING_SLOPE_CLIP = 0.02  # rad: 0.02 で 105km/h 0.35 m/s^2
# GS450h: 実ログで測った線の見え方の揺らぎ (0.1-0.5Hz、舵角から積分した実際の動きとの差) は位置 4.3cm・傾き 1.1-1.5mrad。
# 中心は 22-66m 先で合わせるので、傾きの揺らぎが 44m 先で 5cm の位置ずれになって効く (向きの項だけでは揺れ 3.4→3.1cm)。
# 寄せたいのは一定の片寄りなので、高速の直線では寄せ量 (dk) を 2 秒でなます (→ 2.5/4.6cm、片寄りの残りは 0.5cm)。
# trust には掛けないので、回避・車線変更で手を離す速さは変わらない。
DK_SLOW_TAU = 2.0  # s
LOOKAHEAD_TIME = 1.5  # s
MIN_LOOKAHEAD = 10.0  # m
MIN_SPEED = 5.0  # m/s
LINE_STD = (0.2, 0.6)
POINT_STD = 0.1
YIELD = (0.2, 0.45)
X_IDXS = np.array(ModelConstants.X_IDXS)


class Blendful:
  def __init__(self):
    self.lane_width = FirstOrderFilter(LANE_WIDTH_DEFAULT, 10.0, DT_MDL, initialized=False)
    self.line_gone = FirstOrderFilter(np.zeros(2), 2.0, DT_MDL)
    self.line_probs_slow = FirstOrderFilter(np.zeros(2), 3.0, DT_MDL)
    self.line_probs = FirstOrderFilter(np.zeros(2), 0.5, DT_MDL)
    self.usual_offset = FirstOrderFilter(0.0, 10.0, DT_MDL, initialized=False)
    self.going_around = False
    self.since_turning = 0.0
    self.trust = 0.0
    self.line_trust = 0.0
    self.dk = 0.0 # delta kappa; difference between lane center and model plan
    self.l_prob = 0.0
    self.r_prob = 0.0
    self.model_curvature = 0.0
    self.smooth = 0.0 # smooths the correction target below .53 lat accel
    self.heading = 0.0  # GS450h: 向きの項 (曲率)
    self.dk_slow = 0.0  # GS450h: 高速の直線で dk をなましたもの
    self.curvature = 0.0

  def update(self, model_output: dict[str, np.ndarray], v_ego: float, active: bool, model_curvature: float, enabled: bool = True) -> float:
    self.l_prob = self.r_prob = 0.0
    self.heading = 0.0
    self.model_curvature = model_curvature
    if not enabled:
      if self.curvature or self.trust:
        self.__init__()
      return 0.0
    fading = not active or v_ego < MIN_SPEED
    if fading:
      self.trust = self.line_trust = 0.0
      self.usual_offset.initialized = False
      self.lane_width.initialized = False
      self.going_around = False
      self.dk_slow = 0.0
    target = 0.0 if fading else self._target(model_output, v_ego, model_curvature)
    target = self.smooth = self.smooth + (target - self.smooth) * DT_MDL / (DT_MDL + np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [0.5, 0.0]))
    target += self.heading  # GS450h: 向きの項はなまさない (遅れると揺れを止めずに押す側に回る)
    build = 0.5 / max(v_ego, 1.0) ** 2 * DT_MDL  # jerk over 1 second
    release = (0.5 if fading else 2.0) / max(v_ego, 1.0) ** 2 * DT_MDL  # lets go fast, but past zero it builds like any other pull
    k = self.curvature
    self.curvature = float(np.clip(target, max(k - release, -build) if k > 0 else k - build, min(k + release, build) if k < 0 else k + build))
    return self.curvature

  def _target(self, model_output: dict[str, np.ndarray], v_ego: float, model_curvature: float) -> float:
    lookahead = max(v_ego * LOOKAHEAD_TIME, MIN_LOOKAHEAD)
    lines = model_output['lane_lines'][0, :, :, 0]
    stds = model_output['lane_lines_stds'][0, :, :, 0]

    lone_weight = np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [0.5, 1.0])  # 0.5 on straights, 1.0 in curves
    probs = self.line_probs.update(model_output['lane_lines_prob'][0, 1::2][1:3])
    absent = (probs < 0.3) & (lone_weight > 0.5)
    gone = self.line_gone.update((model_output['lane_lines_prob'][0, 1::2][1:3] < 0.3).astype(float))
    missing = np.where(absent, 1.0, gone)
    dropping = bool(np.any((probs < 0.6 * self.line_probs_slow.update(model_output['lane_lines_prob'][0, 1::2][1:3])) & ~absent))
    l_prob = float(probs[0] * np.interp(np.interp(0.5 * lookahead, X_IDXS, stds[1]), LINE_STD, [1.0, 0.0]))
    r_prob = float(probs[1] * np.interp(np.interp(0.5 * lookahead, X_IDXS, stds[2]), LINE_STD, [1.0, 0.0]))
    self.l_prob, self.r_prob = l_prob, r_prob
    # one line steadily on its own still centers at lone_weight. a line that is disappearing gives control back to laneless
    if missing[1] > 0.8 and l_prob > 0.5:
      lane_trust = l_prob * lone_weight
    elif missing[0] > 0.8 and r_prob > 0.5:
      lane_trust = r_prob * lone_weight
    elif dropping:
      lane_trust = min(l_prob, r_prob)
    else:
      lane_trust = l_prob + r_prob - l_prob * r_prob
    if lone_weight > 0.5 and lane_trust < self.line_trust:
      lane_trust = self.line_trust + (lane_trust - self.line_trust) * DT_MDL / (lone_weight - 0.5 + DT_MDL)
    self.line_trust = lane_trust
    if lane_trust <= 0.0:
      self.trust = 0.0
      return 0.0
    if l_prob + r_prob <= 0.0:  # keep the last fit while trust lets go
      self.trust = min(self.trust, lane_trust)
      return float(self.trust * self.dk_slow)  # GS450h: dk → dk_slow (出している方を保つ)

    win = (X_IDXS >= 0.5 * lookahead) & (X_IDXS <= 1.5 * lookahead)
    x = X_IDXS[win]
    half_width = self.lane_width.x / 2
    width = np.interp(lookahead, X_IDXS, lines[2]) - np.interp(lookahead, X_IDXS, lines[1])
    # GS450h: 幅は両側の線がどちらも中心の計算に効いている時点から見る (l*r > 0.5 を待つと、線が立ち上がる最初の
    # 0.5 秒ほど素通りして真ん中へ引く)。片側が弱いときは強い側の線 + 半車線幅が中心になるので広い道でも外れない。
    wide_scale = float(np.interp(width, WIDE_LANE_FADE, [1.0, 0.0])) if min(l_prob, r_prob) > 0.2 else 1.0
    if l_prob * r_prob > 0.5:
      if wide_scale >= 1.0:  # GS450h: 広い道の幅は「普段の車線幅」として覚えない (片側の線だけのとき目標が右へずれる)
        self.lane_width.update(float(np.clip(width, *LANE_WIDTH_CLIP)))
      half_width = np.clip(lines[2, win] - lines[1, win], *LANE_WIDTH_CLIP) / 2
    l_var = stds[1, win] ** 2 / max(l_prob, 1e-4)
    r_var = stds[2, win] ** 2 / max(r_prob, 1e-4)
    center = (r_var * (lines[1, win] + half_width) + l_var * (lines[2, win] - half_width)) / (l_var + r_var)
    weight = 1.0 / (l_var * r_var / (l_var + r_var) + POINT_STD ** 2)

    # Fit curvature to place the plant in center: e ~ dk x^2 / 2
    plan_x = np.interp(x, model_output['plan'][0, :, 0], model_output['plan'][0, :, 1])
    plan_share = np.interp(abs(model_curvature) * lookahead ** 2 / 2, [0.5, 2.0], [0.3, 0.5]) # up to 50% plan y vs curvature parabola in sharp curves
    error = np.clip(center - ((1 - plan_share) * model_curvature * x ** 2 / 2 + plan_share * plan_x), -1.0, 1.0)
    dk = 2.0 * np.sum(weight * error * x ** 2) / np.sum(weight * x ** 4)

    # A model planning to end up further from its usual spot in the lane than the plant is now is likely going around something
    hw_now = np.clip(lines[2, 0] - lines[1, 0], *LANE_WIDTH_CLIP) / 2 if l_prob * r_prob > 0.5 else self.lane_width.x / 2
    center_now = (l_prob * (lines[1, 0] + hw_now) + r_prob * (lines[2, 0] - hw_now)) / (l_prob + r_prob)
    plan_y = np.interp(lookahead, model_output['plan'][0, :, 0], model_output['plan'][0, :, 1])
    offset = plan_y - np.interp(lookahead, x, center)
    if min(gone) > 0.5:  # both lines gone: turns, intersections, relearn ideal model position
      self.usual_offset.initialized = False
    usual = self.usual_offset.x if self.usual_offset.initialized else offset
    swerve = abs(offset - usual) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2
    away = swerve - abs(-center_now - usual)
    lane_curvature = 2.0 * np.polyfit(x, center, 2)[0]
    inside_curve = abs(lane_curvature) * lookahead ** 2 / 2 > 0.1 and (offset - usual) * lane_curvature > 0
    self.since_turning = 0.0 if abs(model_curvature) * lookahead ** 2 / 2 > 0.5 else self.since_turning + DT_MDL
    if inside_curve:
      self.going_around = False
    elif away > YIELD[0] and self.since_turning > 2.0:
      self.going_around = True
    elif swerve < YIELD[0] / 2:
      self.going_around = False
    self.usual_offset.update_alpha(20.0 if self.going_around else 10.0)
    self.usual_offset.update(offset)

    if inside_curve:
      lane_trust *= np.interp(abs(offset) - abs(center_now) - 0.25 * abs(model_curvature) * lookahead ** 2 / 2, (0.5, 1.0), [1.0, 0.0])
    else:
      lane_trust *= np.interp(swerve, (YIELD[0] / 2, YIELD[0]), [1.0, 0.0]) if self.going_around else np.interp(away, YIELD, [1.0, 0.0])
    lane_trust *= wide_scale

    if lane_trust < self.trust:
      dk = float(np.clip(dk, min(self.dk, 0.0), max(self.dk, 0.0)))
    falling = lane_trust < self.trust
    self.trust = lane_trust if lane_trust < self.trust else self.trust + (lane_trust - self.trust) * DT_MDL / (1.5 + DT_MDL)
    self.dk = dk

    # GS450h: 高速の直線では寄せ量をゆっくりなます (カーブは中心寄せのなましと同じ境目で素通し)。trust が落ちている間は大きくしない。
    straight = np.interp(abs(model_curvature) * v_ego ** 2, [0.18, 0.53], [1.0, 0.0])
    tau = DK_SLOW_TAU * np.interp(v_ego, HEADING_SPEED, [0.0, 1.0]) * straight
    dk_slow = self.dk_slow + (dk - self.dk_slow) * DT_MDL / (DT_MDL + tau)
    self.dk_slow = float(np.clip(dk_slow, min(self.dk_slow, 0.0), max(self.dk_slow, 0.0))) if falling else float(dk_slow)

    # GS450h: 向きの項。各線の近くの傾き (x=0 での 1 次の係数) = 車線に対する向きなので、幅や位置の揺らぎに左右されない。
    # 信頼度ごと落とす (車線変更・回避・広い道では 0)。カーブは中心寄せのなましと同じ境目で切る (確かめたのは直線だけ)。
    near = X_IDXS <= lookahead
    slope = (l_prob * np.polyfit(X_IDXS[near], lines[1, near], 2)[1] + r_prob * np.polyfit(X_IDXS[near], lines[2, near], 2)[1]) / (l_prob + r_prob)
    self.heading = float(self.trust * HEADING_DAMP / v_ego * np.clip(slope, -HEADING_SLOPE_CLIP, HEADING_SLOPE_CLIP)
                         * np.interp(v_ego, HEADING_SPEED, [0.0, 1.0]) * straight)
    return float(self.trust * self.dk_slow)
