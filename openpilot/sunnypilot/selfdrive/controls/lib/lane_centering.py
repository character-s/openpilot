"""
車線中心への幾何補正 (Lane Centering)。StarPilot (firestar5683/StarPilot, MIT) からの移植 (原実装 u/jc01rho)。

e2e の計画経路が両側白線の幾何中心からずれている分を desired_curvature に足す。モデルの判断 (避け・寄せ) は
残し、平衡点だけを中心側へ動かす。白線が怪しい / 車線変更 / ウインカー中は素の e2e へ戻る。

P だけで足りる理由: lookahead L=v 先の横誤差 e(L) ≈ e(0) + ψ·L + κ·L²/2 なので向き (ψ) 項が D として
入る (pure pursuit と同じ安定化) = x=0 で測る純 P だった旧 e_y feedback の微振動が消え、la 換算ゲインは
速度不変で big の復元勾配と同オーダーになる (旧 e_y の P はその 1/3 で、それが I を必要とした)。

制御則での StarPilot 原版との差 = 幅急変ガード (_WIDTH_JUMP_LIMIT、原版に無し) と低速スケジュール
(_MIN_V_EGO / _LOWSPEED_*)、中速スケジュール (_MAX_MIDSPEED_GAIN、既定 x1.5)、高速スケジュール (_HIGHSPEED_*、既定 x2.0)、
片側補完と短い途切れの保持 (_ONE_SIDED_* / _LINE_JUMP_LIMIT / _LOST_HOLD_S、既定 ON)。中速以降と片側補完は
standard / OFF で原版と 1 bit も同じ挙動へ戻せる。ほかに GS 側の追加 = enabled OFF 時の平滑 (_DISABLE_RELEASE_TAU、UI トグルで
走行中に切れるため) と params_fork 経由の設定読み (lane_centering_params.py)。
一度変えた 3 定数 (authority / 幅門 / break_in 帯) は全部原版へ戻した。
定数の採否根拠 = tests/test_lane_centering.py の各 docstring、実測の出所 = archive/probes/_lane_analysis.py。
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


_MIN_V_EGO = 2.5             # [m/s] 渋滞・低速で左に寄るので 5.0 (18km/h) から下げた
_MIN_LANE_PROB = 0.6
_MAX_LANE_STD = 0.3
_MIN_LANE_WIDTH = 2.6        # 原版どおり (2.6m 未満は 1% 台で下げる価値なし)
_MAX_LANE_WIDTH = 4.8
_MAX_OFFSET = 0.3
_MIN_CENTER_TO_LINE = 1.1    # offset を掛けても白線からこれだけは残す (GS 半車幅 0.92m + 余裕)
_MAX_RAW_CORRECTION = 0.004  # [1/m] gain 前の生の曲率上限
_MAX_GAIN = 0.30             # 実効上限 = 0.0012 1/m (GS 換算で舵角 ~3 度) — 45km/h 以上はこの値のまま
# 低速スケジュール: 低速では計画自体が左に寄り、高速向けのゲイン / 不感帯 (0.30 / 0.08) では big の押し返しに負ける。
#   45km/h 以上は良好なのでそこは変えず、29→45km/h で線形に高速側の値へ戻す。_MAX_RAW_CORRECTION は据え置き。
_LOWSPEED_V = (8.0, 12.5)    # [m/s] 29〜45km/h で切り替え
_LOWSPEED_GAIN = 1.0
_LOWSPEED_DEADBAND = 0.04    # [m] (高速側 0.08)
_LOWSPEED_LOOKAHEAD_MIN = 6.0  # [m] (高速側 8.0)

# ★ GS 追加 (09-16): 高速スケジュール。**既定は x2.0 = lcp.DEFAULTS (user 09-16 決定)**。
#   設定で standard (_MAX_GAIN) を選べば 1 bit も変わらない状態へ戻せる = 退避先。
#   09-16 実測: >61km/h の直進は車線中心 +0.005m = 誤差が不感帯 0.08 の内側で LC は何もしていない。
#   ずれているのはカーブで、中心より 0.17-0.26m イン側・エッジ触 3.9-6.9% ⇒ 高速で効かせる先はカーブだけ。
#   ⚠ authority (大きくずれたら譲る) を同じ区間で外すのは、**高速では駐車車両の回避のような
#   「モデルの意思」がほとんど無い**ため (user 09-16)。低速では回避が本物なので触らない。
_HIGHSPEED_V = (17.0, 22.0)        # [m/s] 61〜79km/h で切り替え
_MAX_HIGHSPEED_GAIN = 0.80         # 設定の上限 (lcp.HIGHSPEED_GAIN_CHOICES の最大と一致させる)
_HIGHSPEED_AUTHORITY_SCALE = 0.0   # 79km/h 以上では authority を 0 = 引っ込めない
# ★ GS 追加 (09-24): 高速の不感帯。x2.7 (0.80) にした後のカーブの平衡点は中心から 0.10m イン側で、
#   不感帯 0.08 の床に張り付いている (09-15 の 0.205m → 0.101m、`_lane_analysis.py --only=drift`)。
#   ゲインを上げても 0.08 に漸近するだけ = 効く余地は床を下げる方にしかない。
#   強度に合わせて段階的に下げる (user 10-05): x2.0 (0.60) → 0.06 / x2.7 (0.80) → 0.04、間は線形 (x1.5 = 0.07)。
#   ⚠ ゲイン・authority と同じく「強くする」を選んだときだけ効く (standard なら従来と 1 bit も同じ)。
_HIGHSPEED_DEADBAND_GAINS = (0.30, 0.60, 0.80)   # = (_MAX_GAIN, 既定, _MAX_HIGHSPEED_GAIN)。中速の不感帯もこの表で引く
_HIGHSPEED_DEADBAND = (0.08, 0.06, 0.04)

# ★ GS 追加 (10-10): 中速 (45-61km/h) の強さ。**既定は x1.5 = 様子見 (user 10-10)**、本番候補は x2.0。
#   10-10 実測 (0b8-0c9 の 18 route): 45-61km/h のカーブは LC が効いていても中心より 0.25m イン側・エッジ触 2.9pt
#   (29-45km/h の同条件は +0.11m / 0.9pt)。gain が全帯で最弱の 0.30 で、モデル自身もイン +0.20m を計画している。
#   ⚠ authority (譲る) は**変えない** = 0.5m を超える回避では今までどおり補正 0 (高速と違い回避が本物の帯)。
#   standard を選べば従来と 1 bit も同じ。29→45km/h の低速スケジュールは 1.0 → この値へ下りる形になる。
_MAX_MIDSPEED_GAIN = 0.60

# ★ GS 追加 (10-10): 片側補完。カーブでアウト側の線の std が跳ねて (p50 0.45m、効いている frame は 0.11m)
#   LC が 1.5-3s 止まり、その間に 0.16-0.35m イン側へ寄る (29-61km/h、10-10 実測)。安定しているイン側の線と
#   基準幅から中心を出す。⚠ アウト線の生値は使わない (±0.45m 揺れる) — 基準幅を 1/std² 相当でゆっくり直すのにだけ使う。
#   10-10 検証: 抜けた直後の確かな幅に対して、直前の幅の保持 = 誤差 p50 0.067m / ぶれ線の平均 = 0.11-0.125m (偏りは ±0.02 で無い)。
#   **時間上限は持たない** (user 10-10「途中で抜けるより補い続けた方が分かりやすい」)。抜けるのは根拠が崩れたときだけ。
_ONE_SIDED_MAX_STD = 0.2           # 頼る 1 本は通常の門 (0.3) より厳しく
_ONE_SIDED_GAIN_SCALE = 0.5        # 様子見 (user 10-10)。幅の推定誤差 (中心で 0.03-0.06m) があるので半分の効きから
_ONE_SIDED_DEADBAND_ADD = 0.04     # 推定誤差の p50 ぶんは不感帯で吸う
_ONE_SIDED_ENTRY_MAX_AGE = 3.0     # [s] 補い始めてよいのは、確かな幅 (両側 or 補完中) が直近これ以内にあるときだけ
_ONE_SIDED_UPDATE_MIN_PROB = 0.3   # 基準幅の更新に使うアウト線の下限 (これ未満は線そのものが怪しい)
_ONE_SIDED_UPDATE_MAX_STD = 0.6    # これを超えるアウト線は基準幅の更新にも使わない
_ONE_SIDED_STD_REF = 0.1           # 基準幅の時定数 = _WIDTH_SMOOTH_TAU × (std / これ)²。std 0.2 で 4s、0.45 で 20s
# 補完中は幅急変ガードが使えない (アウト線が信用できない) ので、代わりに頼っている線と計画経路の距離の跳びを見る。
# ⚠ 線の y そのものではなく「線 - 計画経路」を見るのは、カーブ進入で先読み点の y が κL²/2 動く (R100・12m 先で 0.7m) のを
#   相殺するため。イン側に右折レーンが開いて線が逃げる場面 / 回避で経路が動く場面の両方で抜ける。
_LINE_JUMP_LIMIT = 0.50
# 線の確からしさで門から落ちた直後は補正を保持する (0.5s 未満の途切れが 21%、門の切替 0.67 回/s = 10-10 実測)。
# ⚠ 構造変化 (幅の門外れ・幅や線の跳び) では保持しない = 今までどおり即抜く。片側補完と同じトグルで切れる。
_LOST_HOLD_S = 0.3


def _gain_for(v_ego: float, highspeed_gain: float = _MAX_GAIN, midspeed_gain: float = _MAX_GAIN) -> float:
  """低速 (29-45km/h)・中速 (45-61km/h)・高速 (61-79km/h) のスケジュールを 1 本の interp で引く。

  ⚠ np.interp は x が昇順でないと黙って誤った値を返す ⇒ 4 点の並びを崩さないこと。
  """
  mid = max(float(midspeed_gain), _MAX_GAIN)
  return float(np.interp(v_ego,
                         (_LOWSPEED_V[0], _LOWSPEED_V[1], _HIGHSPEED_V[0], _HIGHSPEED_V[1]),
                         (_LOWSPEED_GAIN, mid, mid, max(float(highspeed_gain), _MAX_GAIN))))


def _authority_scale_for(v_ego: float, highspeed_gain: float = _MAX_GAIN) -> float:
  """authority を高速で外す割合。⚠ 「強くする」を選んでいないときは 1.0 = 何もしない。"""
  if highspeed_gain <= _MAX_GAIN:
    return 1.0
  return float(np.interp(v_ego, _HIGHSPEED_V, (1.0, _HIGHSPEED_AUTHORITY_SCALE)))

def _deadband_for(v_ego: float, highspeed_gain: float = _MAX_GAIN, midspeed_gain: float = _MAX_GAIN) -> float:
  """低速 (29-45km/h で 0.04 → 中速の値)、中速 (45-61km/h)、高速 (61-79km/h で → 強度別の値) の 3 段。

  中速・高速とも強度 → 不感帯は同じ表 (_HIGHSPEED_DEADBAND_GAINS) で引く。standard (0.30) は 0.08 = 従来値。
  ⚠ _gain_for と同じく 4 点 1 本の interp = x が昇順でないと np.interp が黙って誤る。
  """
  mid_deadband = float(np.interp(max(float(midspeed_gain), _MAX_GAIN), _HIGHSPEED_DEADBAND_GAINS, _HIGHSPEED_DEADBAND))
  highspeed_deadband = float(np.interp(max(float(highspeed_gain), _MAX_GAIN), _HIGHSPEED_DEADBAND_GAINS, _HIGHSPEED_DEADBAND))
  return float(np.interp(v_ego,
                         (_LOWSPEED_V[0], _LOWSPEED_V[1], _HIGHSPEED_V[0], _HIGHSPEED_V[1]),
                         (_LOWSPEED_DEADBAND, mid_deadband, mid_deadband, highspeed_deadband)))

def _lookahead_min_for(v_ego: float) -> float:
  return float(np.interp(v_ego, _LOWSPEED_V, (_LOWSPEED_LOOKAHEAD_MIN, 8.0)))
_SMOOTH_TAU = 0.4
_SIGNAL_RELEASE_TAU = 0.20
_CONFIDENCE_RELEASE_TAU = 0.20
# ⚠ UI トグルで走行中に enabled を切れるので、reset() (1 frame で 0) だと舵に効いている補正が
# 段差になる。⇒ 他の「抜く」門と同じ 0.2s で平滑する。
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
# 原版どおり。帯を広げても (0.70-0.90 を試した) 平衡点は 2.7cm しか動かず、誤差が大きいときに
# 自然に降りる保護を失う方が高くつく。構造変化 (右折レーン出現) の保護は下の幅ガードが主役。
# 綱引きの計算 = test_break_in_band_is_upstream_default の docstring。
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

# ⚠⚠ 設定の定義と読み書きは `lane_centering_params.py` に集約してある (UI と共有するため)。
# ⚠⚠ openpilot の Params は使わない (理由 = lcp モジュール docstring)。
# ⚠ テストは `lcp.PARAM_DIR` を差し替えるので、参照は必ずモジュール属性経由 (`lcp.xxx`) で行うこと。


class LaneCenteringController:
  def __init__(self, dt: float = 0.01) -> None:
    self._correction = 0.0
    self._width_ref = 0.0
    self.dt = float(dt)
    # 片側補完の状態。⚠ frame は update() のたびに進める (低速・latActive off も含む) = 実時間の物差し。
    self._frame = 0
    self._last_valid_frame: int | None = None     # 確かな幅が最後に得られた frame (両側 or 補完中)
    self._line_rel_ref: list[float | None] = [None, None]   # [左, 右] の「線 - 計画経路」の基準 (τ = _WIDTH_SMOOTH_TAU)
    self._lost_frames = 0
    self._status = 'lost'   # 直近の _raw_correction の結果: ok / one_sided / lost / structural
    # 既定値の出所は lcp.DEFAULTS 1 つ (初回の update_params() が失敗したときの足場もこれ)
    self.enabled = lcp.DEFAULTS[lcp.KEY_ENABLED]
    self.offset = lcp.DEFAULTS[lcp.KEY_OFFSET]
    self.e2e_authority = lcp.DEFAULTS[lcp.KEY_AUTHORITY]
    self.pause_on_signal = lcp.DEFAULTS[lcp.KEY_PAUSE_ON_SIGNAL]
    self.highspeed_gain = lcp.DEFAULTS[lcp.KEY_HIGHSPEED_GAIN]
    self.midspeed_gain = lcp.DEFAULTS[lcp.KEY_MIDSPEED_GAIN]
    self.one_line = lcp.DEFAULTS[lcp.KEY_ONE_LINE]
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
      highspeed_gain = lcp.read_float(lcp.KEY_HIGHSPEED_GAIN)
      midspeed_gain = lcp.read_float(lcp.KEY_MIDSPEED_GAIN)
      one_line = lcp.read_bool(lcp.KEY_ONE_LINE)
    except Exception:
      return
    if not np.isfinite([offset, authority, highspeed_gain, midspeed_gain]).all():
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
    # ⚠ 下限は _MAX_GAIN = 「標準より弱くはしない」(弱める側は低速スケジュールと衝突するため)
    self.highspeed_gain = float(np.clip(highspeed_gain, _MAX_GAIN, _MAX_HIGHSPEED_GAIN))
    self.midspeed_gain = float(np.clip(midspeed_gain, _MAX_GAIN, _MAX_MIDSPEED_GAIN))
    self.one_line = one_line

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
    片側補完の基準 (線と計画経路の距離・最後に確かだった時刻) も同じ理由で一緒に捨てる。
    """
    self._width_ref = 0.0
    self._line_rel_ref = [None, None]
    self._last_valid_frame = None

  def update(self, model_curvature, model_v2, v_ego, lat_active, model_valid,
             lane_change_active=False, turn_signal_active=False) -> float:
    model_curvature = float(model_curvature)
    self._frame += 1

    try:
      v_ego = float(v_ego)
      offset = float(self.offset)
      e2e_authority = float(self.e2e_authority)
      highspeed_gain = float(self.highspeed_gain)
      midspeed_gain = float(self.midspeed_gain)
    except (TypeError, ValueError):
      self.reset()
      return model_curvature

    if not np.isfinite([v_ego, offset, e2e_authority, highspeed_gain, midspeed_gain]).all():
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
      float(np.clip(e2e_authority, 0.0, 1.0)) * _authority_scale_for(v_ego, highspeed_gain),
      _deadband_for(v_ego, highspeed_gain, midspeed_gain),
    )
    hold_frames = int(round(_LOST_HOLD_S / self.dt))
    if not valid:
      # 線の確からしさで落ちた短い途切れは保持する (1 回の途切れにつき 1 度だけ。構造変化では保持しない)
      if self.one_line and self._status == 'lost' and self._lost_frames < hold_frames:
        self._lost_frames += 1
        return model_curvature + self._correction
      self._lost_frames = hold_frames
      # 白線を見失った瞬間に補正を切ると段差になるので、0.2s で抜く
      return model_curvature + self._release(_CONFIDENCE_RELEASE_TAU)
    self._lost_frames = 0

    gain = _gain_for(v_ego, highspeed_gain, midspeed_gain)
    if self._status == 'one_sided':
      gain *= _ONE_SIDED_GAIN_SCALE
    target = float(np.clip(raw_correction, -_MAX_RAW_CORRECTION, _MAX_RAW_CORRECTION)) * gain
    self._correction = float(_smooth(target, self._correction, _SMOOTH_TAU, self.dt))
    return model_curvature + self._correction

  @staticmethod
  def _valid_path(x, y) -> bool:
    return x.size >= 2 and x.size == y.size and np.isfinite(x).all() and np.isfinite(y).all() and np.all(np.diff(x) > 0)

  @staticmethod
  def _covers(x, distance: float) -> bool:
    return bool(x[0] <= distance <= x[-1])

  @staticmethod
  def _line_ok(prob: float, std: float, max_std: float) -> bool:
    return bool(np.isfinite(prob) and np.isfinite(std) and _MIN_LANE_PROB <= prob <= 1.0 and 0.0 <= std <= max_std)

  def _line_at(self, line, lookahead: float) -> float | None:
    """先読み距離での線の y。経路として壊れている / 先読みまで届いていなければ None。"""
    x = np.asarray(line.x, dtype=float)
    y = np.asarray(line.y, dtype=float)
    if not (self._valid_path(x, y) and self._covers(x, lookahead)):
      return None
    return float(np.interp(lookahead, x, y))

  def _raw_correction(self, model_v2, v_ego: float, offset: float, e2e_authority: float,
                      deadband: float | None = None) -> tuple[bool, float]:
    """⚠ 戻り値は (valid, raw) のまま (archive/probes が直接呼ぶ)。落ちた理由・補完かどうかは self._status に残す:
    ok = 両側 / one_sided = 片側補完 / lost = 線の確からしさで落ちた (短い途切れは保持してよい) /
    structural = 幅の門外れ・幅や線の跳び (構造が変わった = 保持せず抜く)。
    """
    self._status = 'lost'
    try:
      lane_lines = model_v2.laneLines
      probs = np.asarray(model_v2.laneLineProbs, dtype=float)
      stds = np.asarray(model_v2.laneLineStds, dtype=float)
      if len(lane_lines) < 3 or probs.size < 3 or stds.size < 3:
        return False, 0.0

      pos_x = np.asarray(model_v2.position.x, dtype=float)
      pos_y = np.asarray(model_v2.position.y, dtype=float)
      # lookahead = v (m) ≒ 1 秒先。ここを見ることで ψ 項が D として入る (docstring 参照)
      lookahead = float(np.clip(v_ego, _lookahead_min_for(v_ego), 35.0))
      if not (self._valid_path(pos_x, pos_y) and self._covers(pos_x, lookahead)):
        return False, 0.0
      model_y = float(np.interp(lookahead, pos_x, pos_y))

      line_ok = [self._line_ok(probs[i], stds[i], _MAX_LANE_STD) for i in (1, 2)]
      lines = [self._line_at(lane_lines[i], lookahead) for i in (1, 2)]
      if deadband is None:
        deadband = _deadband_for(v_ego)

      if all(line_ok) and None not in lines:
        left, right = lines
        width = right - left
        if not _MIN_LANE_WIDTH <= width <= _MAX_LANE_WIDTH:
          # ⚠ ここで基準を捨てないこと。捨てると門内に戻った瞬間に現在幅で初期化されて
          # width_jump = 0 になり、「交差点で一度 5.0m と推定されてから 4.2m に落ち着く」という
          # 構造変化そのものの場面でガードが素通りする。凍結しておけば復帰時に跳びとして出る。
          self._status = 'structural'
          return False, 0.0

        # 幅の急変ガード: 基準幅 (τ1s) から離れている間は譲る。基準は跳んだ後も追従し続けるので
        # 1-2 秒で新しい幅に馴染んで自動復帰する
        if self._width_ref <= 0.0:
          self._width_ref = width
        width_jump = abs(width - self._width_ref)
        self._width_ref = float(_smooth(width, self._width_ref, _WIDTH_SMOOTH_TAU, self.dt))
        if width_jump > _WIDTH_JUMP_LIMIT:
          self._status = 'structural'
          return False, 0.0
        for side in (0, 1):
          rel = lines[side] - model_y
          ref = self._line_rel_ref[side]
          self._line_rel_ref[side] = rel if ref is None else float(_smooth(rel, ref, _WIDTH_SMOOTH_TAU, self.dt))
        status = 'ok'
      else:
        side = self._one_sided_side(probs, stds, line_ok, lines)
        if side is None:
          return False, 0.0
        rel = lines[side] - model_y
        ref = self._line_rel_ref[side]
        if ref is None:
          return False, 0.0
        if abs(rel - ref) > _LINE_JUMP_LIMIT:
          self._status = 'structural'
          return False, 0.0
        self._line_rel_ref[side] = float(_smooth(rel, ref, _WIDTH_SMOOTH_TAU, self.dt))
        self._update_width_from_noisy_line(probs, stds, lines, 1 - side)
        if side == 0:
          left = lines[0]
          right = left + self._width_ref
        else:
          right = lines[1]
          left = right - self._width_ref
        width = self._width_ref
        deadband += _ONE_SIDED_DEADBAND_ADD
        status = 'one_sided'
      self._last_valid_frame = self._frame

      # 狭い車線では offset を自動で縮める (白線から _MIN_CENTER_TO_LINE は必ず残す)
      max_safe_offset = min(_MAX_OFFSET, max(0.0, width * 0.5 - _MIN_CENTER_TO_LINE))
      target_y = 0.5 * (left + right) + float(np.clip(offset, -max_safe_offset, max_safe_offset))
      error = target_y - model_y
      error_abs = abs(error)
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
      self._status = status
      return True, float(2.0 * error / lookahead ** 2)
    except (AttributeError, IndexError, TypeError, ValueError):
      self._status = 'lost'
      return False, 0.0

  def _one_sided_side(self, probs, stds, line_ok, lines) -> int | None:
    """片側補完で頼る線 (0 = 左 / 1 = 右)。補えない場面は None。

    条件: 補完が ON / 一方の線だけが厳しめの門 (std <= _ONE_SIDED_MAX_STD) を通り、もう一方は通常の門すら
    通らない (or 先読みまで届かない) / 基準幅があり、確かな幅が直近 _ONE_SIDED_ENTRY_MAX_AGE 秒以内にある。
    ⚠ 補完中は毎 frame _last_valid_frame が進むので、**一度入れば時間では抜けない** (user 10-10)。
    """
    if not self.one_line or self._width_ref <= 0.0 or self._last_valid_frame is None:
      return None
    if (self._frame - self._last_valid_frame) * self.dt > _ONE_SIDED_ENTRY_MAX_AGE:
      return None
    usable = [line_ok[i] and lines[i] is not None for i in (0, 1)]
    strict = [usable[i] and self._line_ok(probs[i + 1], stds[i + 1], _ONE_SIDED_MAX_STD) for i in (0, 1)]
    for side in (0, 1):
      if strict[side] and not usable[1 - side]:
        return side
    return None

  def _update_width_from_noisy_line(self, probs, stds, lines, other: int) -> None:
    """補完中、ぶれているもう一方の線で基準幅をゆっくり直す (時定数 = τ × (std / 0.1)²)。

    ⚠ 生の線は ±0.45m 揺れるので中心の計算には使わない。平均すれば偏りは無い (10-10 検証 ±0.02m) ので、
    長いカーブで道幅がじわっと変わる分だけをこれで追う。跳び (> _WIDTH_JUMP_LIMIT) や門外の幅は混ぜない。
    """
    y_other = lines[other]
    prob, std = float(probs[other + 1]), float(stds[other + 1])
    if y_other is None or not (np.isfinite(prob) and np.isfinite(std)):
      return
    if prob < _ONE_SIDED_UPDATE_MIN_PROB or not 0.0 <= std <= _ONE_SIDED_UPDATE_MAX_STD:
      return
    left, right = (lines[0], y_other) if other == 1 else (y_other, lines[1])
    width = right - left
    if not _MIN_LANE_WIDTH <= width <= _MAX_LANE_WIDTH or abs(width - self._width_ref) > _WIDTH_JUMP_LIMIT:
      return
    tau = _WIDTH_SMOOTH_TAU * max(1.0, (std / _ONE_SIDED_STD_REF) ** 2)
    self._width_ref = float(_smooth(width, self._width_ref, tau, self.dt))
