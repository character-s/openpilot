"""
車線中心への幾何補正 (Lane Centering)。

StarPilot (firestar5683/StarPilot, MIT) の `selfdrive/controls/lib/lane_centering.py` を
GS 450h 向けに移植 (2026-08-26)。原実装 = u/jc01rho。

役割 = e2e モデルの計画経路が「両側白線の幾何中心」からずれている分を desired_curvature に
足して埋め合わせる。モデルの判断 (避け・寄せ) はそのまま残し、平衡点だけを中心側へ動かす。
白線が怪しい / 車線変更中 / ウインカー中は即座に素の e2e へ戻る。

★ 08-26 の実測 (route 162 + 165、TT big、195k frame、`_lane_analysis.py --only=position`):
  - カーブの plan_in p50 = **+0.22m (162) / +0.26m (165)**
  - 直進の「計画 vs 実測」乖離 = **-0.009 / -0.007m** ⇒ 車は計画どおり走っている。
    **イン寄りは追従誤差ではなく計画自体** = だからこの層 (計画の補正) で叩くのが正しい
  - エッジ接触 = 直進 2.4% / カーブ 11.1%、**幅 <2.9m のカーブでは 31.8%**

★ なぜ P だけで足りるか (旧 e_y feedback は I が必要だった):
  1. lookahead L 先の横誤差は e(L) ≈ e(0) + ψ·L + κ·L²/2 なので **ψ (向き) 項が自動的に D 項
     になる** (L = v ⇒ D ゲイン = v) = pure pursuit と同じ安定化。x=0 で測る純 P だった
     e_y v2 の微振動 (「サイドミラーで分かる程度」) はこれが無かったため
  2. la 換算のゲインは 8-35 m/s で `0.6 × (|err| - 0.08)` と**速度不変**になり (lookahead=v なので
     v² が約分される)、big の復元勾配 g = 0.0017-0.0035 /m と**同オーダー**。綱引き
     `d1 = (g·d0 + c·dz)/(g+c)` で平衡点が 0.40 → 0.22m まで動く。旧 e_y の P は g の 1/3 しか
     なく、それが I を必要とした理由だった
  詳細 = memory `project_gs_lateral_nlcc` の「⏭ 方針」節。

GS 向けの変更点 (StarPilot 原版との差は 2 つだけ):
  **調整した定数は 1 つもない** — 08-26 に 3 つ (authority=0 / 幅門 2.4 / break_in 帯) を変えたが、
  実測と綱引きの計算で**全部「変える価値なし」と判定して原版へ戻した**:
    - `E2EAuthority = 0` → ✗ 構造変化の保護が消える (幅 3.0→4.0m で la +0.234 m/s² = 引き込まれ)
    - `_MIN_LANE_WIDTH` 2.4 → ✗ 幅 p01 = 2.59m / 2.6m 未満は 1.14% = 下げても拾えない
    - break_in 帯 0.70-0.90 → ✗ 平衡点は原版 0.245m / 移動版 0.218m で**差 2.7cm** (上の定数コメント)
  唯一の追加が **車線幅の急変ガード** (原版になし、閾値 0.50m = 実測 |Δ幅| 1 秒窓 p99 0.513m の
  直上 ⇒ 発動は約 1%)。原版の break_in は誤差の大きさしか見ないので、「幅が広がったが誤差は
  0.5m 未満」の場面だけ素通りする。そこを塞ぐ最小の安全網で、通常走行の挙動はほぼ変えない。
"""
import numpy as np

from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering_params as lcp


def _smooth(val, prev_val, tau, dt):
  """selfdrive/controls/lib/drive_helpers.py の smooth_value と同一。

  ここに複製しているのは依存を numpy だけに保つため — drive_helpers も common.realtime も
  cereal (→ opendbc submodule) を引くので、import すると PC 側で純粋な単体テストが回せなくなる。
  """
  alpha = 1 - np.exp(-dt / tau) if tau > 0 else 1
  return alpha * val + (1 - alpha) * prev_val


_MIN_V_EGO = 2.5             # 09-02: 5.0 (18km/h) → 2.5 (9km/h)。渋滞・低速で左に寄るため (memory lateral_nlcc 09-02)
_MIN_LANE_PROB = 0.6
_MAX_LANE_STD = 0.3
_MIN_LANE_WIDTH = 2.6        # ⚠ 原版どおり。08-26 実測で幅 p01 = 2.59m / 2.6m 未満は 1.14% しか
                             # ないと分かったので、2.4 へ下げても拾えるのは 1% 台 = 変える理由なし
_MAX_LANE_WIDTH = 4.8
_MAX_OFFSET = 0.3
_MIN_CENTER_TO_LINE = 1.1    # offset を掛けても白線からこれだけは残す (GS 半車幅 0.92m + 余裕)
_MAX_RAW_CORRECTION = 0.004  # [1/m] gain 前の生の曲率上限
_MAX_GAIN = 0.30             # 実効上限 = 0.0012 1/m (GS 換算で舵角 ~3 度) — 45km/h 以上はこの値のまま
# ★ 09-02 低速スケジュール (user「低速で左に寄る」)。IDM は <18km/h で計画自体が +0.12〜0.18m 左で、
#   ゲイン 0.30 / 不感帯 0.08 では 29-45km/h でも計画と実測が同じ = big の押し返しに負けていた。
#   45km/h 以上は中心 ±2cm で良好なのでそこは変えず、29→45km/h で線形に現行値へ戻す。
#   補正曲率の絶対上限 _MAX_RAW_CORRECTION は据え置き (20km/h で横 G 0.12 m/s²)。
_LOWSPEED_V = (8.0, 12.5)    # [m/s] 29〜45km/h で切り替え
_LOWSPEED_GAIN = 1.0
_LOWSPEED_DEADBAND = 0.04    # [m] (現行 0.08)
_LOWSPEED_LOOKAHEAD_MIN = 6.0  # [m] (現行 8.0)

def _gain_for(v_ego: float) -> float:
  return float(np.interp(v_ego, _LOWSPEED_V, (_LOWSPEED_GAIN, _MAX_GAIN)))

def _deadband_for(v_ego: float) -> float:
  return float(np.interp(v_ego, _LOWSPEED_V, (_LOWSPEED_DEADBAND, _CENTER_ERROR_DEADBAND)))

def _lookahead_min_for(v_ego: float) -> float:
  return float(np.interp(v_ego, _LOWSPEED_V, (_LOWSPEED_LOOKAHEAD_MIN, 8.0)))
_SMOOTH_TAU = 0.4
_SIGNAL_RELEASE_TAU = 0.20
_CONFIDENCE_RELEASE_TAU = 0.20
# ⚠ 08-26 追加: UI トグル (mici の設定画面) を足したことで **走行中に enabled を切れる**
# ようになった。それまで enabled は起動時に決まる値でしかなく、reset() で 1 frame で 0 に
# 落としても実害が無かったが、今は舵に効いている補正をユーザー操作で消すことになるので
# 段差が出る。⇒ 他の「抜く」門と同じ 0.2s で平滑する。
_DISABLE_RELEASE_TAU = 0.20
# 平滑は指数減衰なので厳密には 0 にならない。これ以下は 0 とみなして計算から降りる
# (既定 OFF のユーザーが毎 frame exp を踏まないため)。実効上限 0.0012 1/m の 1/1000 以下。
_RELEASE_EPS = 1e-6
_CENTER_ERROR_DEADBAND = 0.08

# ⚠ 08-26 実測 (`archive/probes/_lane_centering_design.py`、162+165 で n=30,457): yStd は
# **p50 0.024 / p95 0.038m** で、この閾値 0.35 を下回る frame が **100.0%**。つまりこの門は
# 実質いつも真で、authority は「常時効くゲート」になる (原版が想定した「モデルが迷っている
# ときは authority を効かせない」という分岐は TT では発生しない)。⇒ 帯の置き方が全て。
_E2E_MAX_PATH_STD = 0.35
# ⚠ 原版どおり。08-26 に誤差帯ごとのカーブ率を測ると (全体 12.6%)、0.0-0.2m:5.8% / 0.2-0.4:19.4% /
# 0.4-0.6:52.9% / 0.6-0.8:79.5% / 0.8m超:9.7% で、「大きい誤差 = モデルが避けている」という原版の
# 前提は 0.8m まで成り立たない (むしろカーブのイン寄り = 直したい対象) ⇒ 一度 0.70-0.90 へ移した。
# ⚠⚠ **が、綱引きを解くと移す価値がほとんど無い**: 補正が山型なので平衡点は「山の内側」に
# 引き寄せられ、`0.455×(0.40−d) = f(d)` の解は **原版 0.245m / 移動版 0.218m = 差 2.7cm**
# (元の平衡点 0.40m に対し −39% vs −45%)。2.7cm のために「0.5m 超の誤差でも効き続ける」リスクを
# 買うのは割に合わない。⇒ **原版の帯に戻した**。原版の帯は誤差が大きいと自然に降りるので、
# 車線構造が変わった場面の保護も兼ねている。
# ⚠ 構造変化 (右折レーン出現など) の保護は帯ではなく下の幅ガードが主役 — 誤差の大きさだけでは
# 「深いイン寄り」と「車線が広がった」を区別できないため。
_E2E_BREAK_IN_START = 0.15
_E2E_BREAK_IN_FULL = 0.50

# ★ GS 追加: 車線幅の急変ガード。break_in は path_std <= 0.35 のときしか効かない (モデルが
# 迷っていると素通りする) ので、幅そのものの跳びを独立に見る。右折レーンの出現・分岐・停車帯で
# 中心が一気に動く場面を「構造が変わった」として補正を抜き、新しい幅に馴染むまで待つ。
_WIDTH_SMOOTH_TAU = 1.0      # [s] 基準幅の時定数
_WIDTH_JUMP_LIMIT = 0.50     # [m] 08-26 実測 (seg 境界を跨ぐ差分を除いた値): |Δ幅| の 1 秒窓は
                             # p50 0.081 / p95 0.321 / p99 0.513m。0.35 だと通常の揺らぎで 3.84%
                             # の frame が抜ける。0.50 = p99 の直上で、発動は約 1%。
                             # 右折レーン出現級 (実測 max 1.718m) は確実に捕まえる

PARAM_UPDATE_FRAMES = 100    # controlsd が update_params() を呼ぶ間隔 [frame]。100Hz なので 1Hz

# ⚠⚠ 設定 4 つの定義と読み書きは `lane_centering_params.py` に集約してある (UI と共有するため)。
# ⚠⚠ openpilot の Params は使わない。`/data/params` に置くと `clearAll` のホワイトリスト
# (= `libparams_c.so` に焼かれた表) に無いキーとして **manager 起動のたびに消される**ため。
# 保存先は `/data/params_fork/d` (理由の全文は lane_centering_params.py の docstring)。
# ⚠ テストは `lcp.PARAM_DIR` を差し替えるので、参照は必ずモジュール属性経由 (`lcp.xxx`) で行うこと。


class LaneCenteringController:
  def __init__(self, dt: float = 0.01) -> None:
    self._correction = 0.0
    self._width_ref = 0.0
    self.dt = float(dt)
    self.enabled = False
    self.offset = 0.0
    self.e2e_authority = 1.0
    self.pause_on_signal = True
    self.update_params()

  def update_params(self) -> None:
    """1Hz で読む。トグルを走行中に切れることが安全上の要 (UI から即 OFF できる)。"""
    # ⚠ 一時変数に全部読んでから一括で反映する。途中で失敗したときに「enabled だけ新しい値、
    # offset は古い値」という不整合な組み合わせで走らせないため。
    # ⚠ 例外は広く握る — ここは controlsd (安全上クリティカル) の中なので、params 層の
    # どんな失敗も制御ループを落としてはいけない。失敗時は前回値のまま走る。
    try:
      enabled = lcp.read_bool(lcp.KEY_ENABLED)
      offset = lcp.read_float(lcp.KEY_OFFSET)
      authority = lcp.read_float(lcp.KEY_AUTHORITY)
      pause_on_signal = lcp.read_bool(lcp.KEY_PAUSE_ON_SIGNAL)
    except Exception:
      return
    if not np.isfinite([offset, authority]).all():
      return
    if enabled and not self.enabled:
      # ⚠ OFF→ON: OFF の間は _raw_correction を通らないので _width_ref が更新されず、
      # 「切った場所の車線幅」が残ったままになる。別の幅の道路で ON に戻すと width_jump が
      # 閾値を超えて幅急変ガードが誤発動し、ON にした直後 (= 効きを確かめたい場面) だけ
      # 1-2 秒補正が出ない。⇒ 基準を捨てて復帰後の最初のフレームの幅で取り直す。
      self._reset_width_reference()
    self.enabled = enabled
    self.offset = float(np.clip(offset, -_MAX_OFFSET, _MAX_OFFSET))
    self.e2e_authority = float(np.clip(authority, 0.0, 1.0))
    self.pause_on_signal = pause_on_signal

  def apply(self, desired_curvature, sm, CS, CC, maneuver_active: bool, lane_change_active: bool):
    """controlsd から 1 行で呼ぶための入口。

    ⚠ ここに集約しているのは **upstream との差分を controlsd.py の 4 行に抑えるため**。
    sunnypilot の `ControlsExt` には desired_curvature を加工するフックが無く、かつ
    `controlsd_ext.py` は upstream 側のファイルなので触ると追従で衝突する。⇒ 呼び出し口を
    こちら (新規ファイル = 衝突しない) に置き、controlsd.py には呼び出しだけを残す。

    maneuver_active = `lateralManeuverPlan` が valid。操舵マニューバの測定を汚さないよう
    補正は入れず、状態も持ち越さない (明けに古い _correction から平滑が再開しないように)。

    ⚠ lane_change の判定は**呼び出し側で**やって bool で渡す。ここで `log.LaneChangeState` を
    参照すると cereal (→ opendbc submodule) を引いてしまい、この入口が PC で単体テストできなく
    なるため。controlsd は元々 `LaneChangeState` を import しているので差分は増えない。
    """
    if sm.frame % PARAM_UPDATE_FRAMES == 0:
      self.update_params()
    if maneuver_active:
      self.reset()
      return desired_curvature
    return self.update(desired_curvature, sm['modelV2'], CS.vEgo, CC.latActive,
                       bool(sm.all_checks(['modelV2'])), lane_change_active,
                       bool(CS.leftBlinker or CS.rightBlinker))

  def reset(self) -> None:
    self._correction = 0.0

  def _release(self, tau: float) -> float:
    """補正を tau で 0 へ抜き、抜いた後の補正値を返す。

    ⚠ `reset()` (= 1 frame で 0) との使い分け:
      - `_release()` = **舵に効いている補正を消す**とき (ユーザーが切った / ウインカー /
        白線ロスト)。1 frame で落とすと段差になる。
      - `reset()` = そもそも補正が舵に出ていない場面 (latActive off / 低速 / モデル停止 /
        車線変更)。ここで平滑しても意味がなく、状態を持ち越す方が有害。
    """
    if self._correction == 0.0:
      return 0.0
    self._correction = float(_smooth(0.0, self._correction, tau, self.dt))
    if abs(self._correction) < _RELEASE_EPS:
      self._correction = 0.0
    return self._correction

  def _reset_width_reference(self) -> None:
    """幅の基準を捨てる。⚠ 呼ぶのは **車線変更** と **OFF→ON** の 2 つだけ。

    停車 (v<5) や latActive off でも捨てると、発進や再 engage のたびに基準が現在幅で
    初期化され、直後に幅が変わっても「跳び」として検出できなくなる。低速の交差点まわりは
    まさに車線構造が変わる場所なので、ガードが最も要る区間で無効化されてしまう。
    ⇒ 捨ててよいのは「基準が確実に無効になった」ときだけ = 車線が変わった / OFF の間に
    基準の更新が止まっていた、の 2 つ。
    """
    self._width_ref = 0.0

  def update(self, model_curvature, model_v2, v_ego, lat_active, model_valid,
             lane_change_active=False, turn_signal_active=False) -> float:
    model_curvature = float(model_curvature)

    try:
      v_ego = float(v_ego)
      offset = float(self.offset)
      e2e_authority = float(self.e2e_authority)
    except (TypeError, ValueError):
      self.reset()
      return model_curvature

    if not np.isfinite([v_ego, offset, e2e_authority]).all():
      self.reset()
      return model_curvature

    # ⚠ この 3 つは「補正が舵に出ていない」場面なので即断でよい (_release ではなく reset)
    if not model_valid or not lat_active or v_ego < _MIN_V_EGO:
      self.reset()
      return model_curvature

    # ⚠ enabled OFF だけは平滑して抜く。UI から走行中に切れるようになったので、ここで
    # reset() すると「切った瞬間に舵が跳ねる」= 一番やってはいけない切り方になる。
    if not self.enabled:
      return model_curvature + self._release(_DISABLE_RELEASE_TAU)

    if self.pause_on_signal and turn_signal_active:
      return model_curvature + self._release(_SIGNAL_RELEASE_TAU)

    if lane_change_active:
      self.reset()
      self._reset_width_reference()   # 車線が変われば幅の基準は無効
      return model_curvature

    valid, raw_correction = self._raw_correction(
      model_v2,
      v_ego,
      float(np.clip(offset, -_MAX_OFFSET, _MAX_OFFSET)),
      float(np.clip(e2e_authority, 0.0, 1.0)),
    )
    if not valid:
      # 白線を見失った瞬間に補正を切ると段差になるので、0.2s で抜く
      return model_curvature + self._release(_CONFIDENCE_RELEASE_TAU)

    target = float(np.clip(raw_correction, -_MAX_RAW_CORRECTION, _MAX_RAW_CORRECTION)) * _gain_for(v_ego)
    self._correction = float(_smooth(target, self._correction, _SMOOTH_TAU, self.dt))
    return model_curvature + self._correction

  @staticmethod
  def _valid_path(x, y) -> bool:
    return x.size >= 2 and x.size == y.size and np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.diff(x) > 0)

  @staticmethod
  def _covers(x, distance: float) -> bool:
    return bool(x[0] <= distance <= x[-1])

  def _raw_correction(self, model_v2, v_ego: float, offset: float, e2e_authority: float) -> tuple[bool, float]:
    try:
      lane_lines = model_v2.laneLines
      probs = np.asarray(model_v2.laneLineProbs, dtype=float)
      stds = np.asarray(model_v2.laneLineStds, dtype=float)
      if len(lane_lines) < 3 or probs.size < 3 or stds.size < 3:
        return False, 0.0
      if not np.isfinite(probs[[1, 2]]).all() or not np.isfinite(stds[[1, 2]]).all():
        return False, 0.0
      if np.any(probs[[1, 2]] < _MIN_LANE_PROB) or np.any(probs[[1, 2]] > 1.0):
        return False, 0.0
      if np.any(stds[[1, 2]] < 0.0) or np.any(stds[[1, 2]] > _MAX_LANE_STD):
        return False, 0.0

      left_x = np.asarray(lane_lines[1].x, dtype=float)
      left_y = np.asarray(lane_lines[1].y, dtype=float)
      right_x = np.asarray(lane_lines[2].x, dtype=float)
      right_y = np.asarray(lane_lines[2].y, dtype=float)
      pos_x = np.asarray(model_v2.position.x, dtype=float)
      pos_y = np.asarray(model_v2.position.y, dtype=float)
      if not (self._valid_path(left_x, left_y) and self._valid_path(right_x, right_y) and self._valid_path(pos_x, pos_y)):
        return False, 0.0

      # lookahead = v (m) ≒ 1 秒先。ここを見ることで ψ 項が D として入る (docstring 参照)
      lookahead = float(np.clip(v_ego, _lookahead_min_for(v_ego), 35.0))
      if not all(self._covers(x, lookahead) for x in (left_x, right_x, pos_x)):
        return False, 0.0

      left = float(np.interp(lookahead, left_x, left_y))
      right = float(np.interp(lookahead, right_x, right_y))
      width = right - left
      if not _MIN_LANE_WIDTH <= width <= _MAX_LANE_WIDTH:
        # ⚠ ここで基準を捨てないこと。捨てると門内に戻った瞬間に現在幅で初期化されて
        # width_jump = 0 になり、「交差点で一度 5.0m と推定されてから 4.2m に落ち着く」という
        # 構造変化そのものの場面でガードが素通りする。凍結しておけば復帰時に跳びとして出る。
        return False, 0.0

      # 幅の急変ガード: 基準幅 (τ1s) から離れている間は譲る。基準は跳んだ後も追従し続けるので
      # 1-2 秒で新しい幅に馴染んで自動復帰する
      if self._width_ref <= 0.0:
        self._width_ref = width
      width_jump = abs(width - self._width_ref)
      self._width_ref = float(_smooth(width, self._width_ref, _WIDTH_SMOOTH_TAU, self.dt))
      if width_jump > _WIDTH_JUMP_LIMIT:
        return False, 0.0

      # 狭い車線では offset を自動で縮める (白線から _MIN_CENTER_TO_LINE は必ず残す)
      max_safe_offset = min(_MAX_OFFSET, max(0.0, width * 0.5 - _MIN_CENTER_TO_LINE))
      target_y = 0.5 * (left + right) + float(np.clip(offset, -max_safe_offset, max_safe_offset))
      model_y = float(np.interp(lookahead, pos_x, pos_y))
      error = target_y - model_y
      error_abs = abs(error)
      deadband = _deadband_for(v_ego)
      if error_abs <= deadband:
        error = 0.0
      else:
        error = np.copysign(error_abs - deadband, error)

      # e2e authority: モデルが自信を持って (path std が小さい) 大きく外している = 障害物回避の
      # 可能性があるので補正を譲る。⚠ 既定は原版どおり 1.0 = 譲る (0 にすると構造変化の保護が
      # 消えることを 08-26 に実測: 幅 3.0→4.0m で la +0.234 m/s² = 広がった側へ引き込まれた)
      try:
        pos_y_std = np.asarray(model_v2.position.yStd, dtype=float)
        if self._valid_path(pos_x, pos_y_std):
          path_std = float(np.interp(lookahead, pos_x, pos_y_std))
          if 0.0 <= path_std <= _E2E_MAX_PATH_STD:
            break_in = np.clip(
              (error_abs - _E2E_BREAK_IN_START) / (_E2E_BREAK_IN_FULL - _E2E_BREAK_IN_START),
              0.0,
              1.0,
            )
            error *= 1.0 - e2e_authority * float(break_in)
      except (AttributeError, TypeError, ValueError):
        pass

      # y = κ·x²/2 の逆 = 「lookahead 先で error だけ横に動くのに要る曲率」
      return True, float(2.0 * error / lookahead ** 2)
    except (AttributeError, IndexError, TypeError, ValueError):
      return False, 0.0
