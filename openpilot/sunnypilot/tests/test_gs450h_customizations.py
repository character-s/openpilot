"""upstream 追従で GS 450h 改造が黙って落ちていないかを検出する。

⚠ これは機能テストではない。**追従作業の事故を検出するテスト**。個々の値が正しいかは
実走で決めたことであり、ここが守るのは「**実走で決めた値が今もそこに在るか**」だけ。

sunnypilot 本家 `staging-chestnut` は orphan force-push なので、追従は毎回
「新ベースを checkout して GS 改造を cherry-pick で載せ直す」になる。そこで起きる事故が 4 種類:

1. **改造が載り漏れる** — 定数が upstream の値に戻る。⚠ 走らせるまで気づけない (静かに劣化する)
2. **上流が API を消す/改名する** — GS 側が呼んでいる関数が無くなり、実車で ImportError
3. **捨てた関数への参照が残る** — 逆方向。定義だけ消して呼び出し元が残る
4. **car 側と panda 側がズレる** — ⚠⚠ 操舵の限界値は 2 箇所に書いてある。片方だけ載ると
   panda が tx を拒否して**操舵が効かなくなる**

⚠ import は一切しない (openpilot の import は依存が重く、CI で回すと scons が要る)。
ソースを `ast` と正規表現で読んで、値がそこに在るかだけを見る。⇒ 依存は pytest だけ。

⚠⚠ **値を変えたくなったら、まず実走で確かめてからここを直すこと。**
テストを通すために値を書き換えるのは、実走で決めた根拠を捨てるのと同じ。
根拠は各定数のコメントと memory (`project_gs_steer_headroom` / `project_gs_longitudinal_tune` 等) にある。
"""

import ast
import re
from pathlib import Path

import pytest

# openpilot/sunnypilot/tests/ から数えて 3 つ上がリポジトリのルート
REPO_ROOT = Path(__file__).resolve().parents[3]

VALUES_PY = 'opendbc_repo/opendbc/car/toyota/values.py'
INTERFACE_PY = 'opendbc_repo/opendbc/car/toyota/interface.py'
SAFETY_H = 'opendbc_repo/opendbc/safety/modes/toyota.h'
CARCONTROLLER_PY = 'opendbc_repo/opendbc/car/toyota/carcontroller.py'
LONGCONTROL_PY = 'openpilot/selfdrive/controls/lib/longcontrol.py'
DRIVE_HELPERS_PY = 'openpilot/selfdrive/controls/lib/drive_helpers.py'
LONG_MPC_PY = 'openpilot/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py'
MODELD_PY = 'openpilot/sunnypilot/modeld_v2/modeld.py'
NNLC_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/nnlc/nnlc.py'
TORQUE_EXT_BASE_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext_base.py'
SCC_VISION_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/vision_controller.py'
SCC_MAP_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/smart_cruise_control/map_controller.py'
LC_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/lane_centering.py'
LC_PARAMS_PY = 'openpilot/sunnypilot/selfdrive/controls/lib/lane_centering_params.py'
MICI_MODELS_PY = 'openpilot/selfdrive/ui/sunnypilot/mici/layouts/models.py'
MICI_TOGGLES_PY = 'openpilot/selfdrive/ui/mici/layouts/settings/toggles.py'
PARAMS_KEYS_H = 'openpilot/common/params_keys.h'


# ---------------------------------------------------------------------------
# ソースを読むための道具 (import はしない)
# ---------------------------------------------------------------------------

def _read(rel: str) -> str:
  path = REPO_ROOT / rel
  if not path.exists():
    pytest.fail(f"{rel} が無い。上流がファイルごと移動/削除した可能性がある")
  return path.read_text(encoding='utf-8')


def _toplevel_names(rel: str) -> set[str]:
  """モジュールのトップレベルで定義されている名前を集める。"""
  names: set[str] = set()
  for node in ast.parse(_read(rel), filename=rel).body:
    if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
      names.add(node.name)
    elif isinstance(node, ast.Assign):
      names.update(t.id for t in node.targets if isinstance(t, ast.Name))
    elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
      names.add(node.target.id)
  return names


def _literal(rel: str, name: str):
  """トップレベルの `name = <リテラル>` を取り出す。定数が upstream の値に戻ると落ちる。"""
  for node in ast.parse(_read(rel), filename=rel).body:
    targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, ast.AnnAssign) else [])
    for target in targets:
      if isinstance(target, ast.Name) and target.id == name:
        try:
          return ast.literal_eval(node.value)
        except ValueError:
          pytest.fail(f"{rel}: {name} がリテラルでなくなっている ({ast.unparse(node.value)[:70]})")
  pytest.fail(f"{rel}: {name} が定義されていない。追従で載り漏れた疑い")


def _assignments_in_gs_f_branch(rel: str, attr_root: str) -> dict[str, object]:
  """`if ... LEXUS_GS_F ...:` ブロックの中の `<attr_root>.X.Y = <リテラル>` を集める。

  ⚠ GS 改造は「LEXUS_GS_F のときだけ値を差し替える」形で入っているので、
  トップレベル定数と違ってブロックごと消える落ち方をする。ブロック自体の存在も見たい。
  """
  found: dict[str, object] = {}
  for node in ast.walk(ast.parse(_read(rel), filename=rel)):
    if not isinstance(node, ast.If) or 'LEXUS_GS_F' not in ast.unparse(node.test):
      continue
    for sub in ast.walk(node):
      if not isinstance(sub, ast.Assign):
        continue
      for target in sub.targets:
        if not isinstance(target, ast.Attribute):
          continue
        chain = ast.unparse(target)
        if not chain.startswith(f'{attr_root}.'):
          continue
        try:
          found[chain.split('.')[-1]] = ast.literal_eval(sub.value)
        except ValueError:
          continue
  if not found:
    pytest.fail(f"{rel}: LEXUS_GS_F の分岐で {attr_root}.* への代入が見つからない。ブロックごと落ちた疑い")
  return found


def _c_struct_fields(rel: str, struct_name: str) -> dict[str, int]:
  """panda 安全側の C 構造体から `.field = N` を取り出す。

  ⚠ C をパースするのではなく、`= {` から対応する `}` までを切り出して指定子を拾うだけ。
  安全側の値は数値リテラル直書きなので、これで十分かつ壊れにくい。
  """
  src = _read(rel)
  start = src.find(struct_name)
  if start < 0:
    pytest.fail(f"{rel}: 構造体 {struct_name} が無い。panda 側の GS 改造が落ちている")
  brace = src.find('{', start)
  end = src.find('};', brace)
  if brace < 0 or end < 0:
    pytest.fail(f"{rel}: {struct_name} の本体を切り出せない")
  body = src[brace:end]
  fields = {m.group(1): int(m.group(2)) for m in re.finditer(r'\.(\w+)\s*=\s*(-?\d+)', body)}
  if not fields:
    pytest.fail(f"{rel}: {struct_name} に数値フィールドが無い")
  return fields


# ===========================================================================
# 1) 車両制御 (opendbc) — ⚠ 安全に直結。ここが崩れると操舵が効かなくなる
# ===========================================================================

# ⚠ 値の根拠は values.py のコメントと [[project_gs_steer_headroom]] にある。実走で決めた値。
GS_F_STEER_LIMITS = {
  'STEER_MAX': 1500,                    # EPS の hard limit。これ以上は EPS が受け付けない
  'STEER_ERROR_MAX': 900,               # Stage 4 (07-16): 750 -> 900
  'STEER_DELTA_UP': 20,                 # Stage 9 (08-15): 15 -> 20。低速急カーブの立ち上がり短縮
  'STEER_DELTA_DOWN': 45,               # Stage 8 (08-03): 高トルク帯は fault 実績のある 45 に戻す
  'STEER_DELTA_DOWN_FAST': 50,          # 低トルク帯だけ 50
  'STEER_DELTA_DOWN_FAST_BELOW': 500,   # fault 最小事例 741 の下に余裕を取った閾値
}


@pytest.mark.parametrize(('name', 'expected'), sorted(GS_F_STEER_LIMITS.items()))
def test_gs_f_raised_steer_limits_survive(name, expected):
  actual = _assignments_in_gs_f_branch(VALUES_PY, 'self').get(name)
  assert actual == expected, f"{name} が {expected} から {actual} に変わっている (実走で決めた値)"


def test_panda_gs_f_limits_struct_survives():
  """panda 側の GS 専用リミット構造体。⚠ 消えると upstream の狭いリミットが使われる。"""
  fields = _c_struct_fields(SAFETY_H, 'TOYOTA_GS_F_TORQUE_STEERING_LIMITS')
  assert fields.get('max_torque') == 1800
  assert fields.get('max_rate_up') == 25
  assert fields.get('max_rate_down') == 50
  assert fields.get('max_torque_error') == 900
  assert fields.get('max_rt_delta') == 1500


def test_car_side_never_exceeds_panda_side():
  """⚠⚠ **これが一番大事なテスト。**

  操舵の限界値は car 側 (values.py) と panda 側 (toyota.h) の 2 箇所に書いてある。
  car 側が panda 側を 1 でも超えると、panda が tx を弾いて**操舵が出なくなる**。
  片方だけ cherry-pick された瞬間にこうなるので、値そのものより**関係**を守る。
  """
  car = _assignments_in_gs_f_branch(VALUES_PY, 'self')
  panda = _c_struct_fields(SAFETY_H, 'TOYOTA_GS_F_TORQUE_STEERING_LIMITS')

  assert car['STEER_MAX'] <= panda['max_torque'], \
    f"car の STEER_MAX {car['STEER_MAX']} > panda の max_torque {panda['max_torque']} = panda が弾く"
  assert car['STEER_DELTA_UP'] <= panda['max_rate_up'], \
    f"car の DELTA_UP {car['STEER_DELTA_UP']} > panda の max_rate_up {panda['max_rate_up']} = panda が弾く"
  assert car['STEER_DELTA_DOWN'] <= panda['max_rate_down'], \
    f"car の DELTA_DOWN {car['STEER_DELTA_DOWN']} > panda の max_rate_down {panda['max_rate_down']}"
  assert car['STEER_DELTA_DOWN_FAST'] <= panda['max_rate_down'], \
    f"car の DELTA_DOWN_FAST {car['STEER_DELTA_DOWN_FAST']} > panda の max_rate_down {panda['max_rate_down']}"
  assert car['STEER_ERROR_MAX'] <= panda['max_torque_error'], \
    f"car の ERROR_MAX {car['STEER_ERROR_MAX']} > panda の max_torque_error {panda['max_torque_error']}"

  # max_rt_delta は 250ms (25 frames) 分の累積上限。car 側が出せる最速で埋めても超えないこと
  worst_case = 25 * car['STEER_DELTA_DOWN_FAST']
  assert worst_case <= panda['max_rt_delta'], \
    f"250ms 累積 {worst_case} > panda の max_rt_delta {panda['max_rt_delta']} = 連続クリップで弾かれる"


def test_raised_steer_limits_flag_bit_matches_between_car_and_panda():
  """⚠ フラグのビット位置が car と panda で一致していないと、リミットが切り替わらない。

  car: `RAISED_STEER_LIMITS = (16 << 8)` / panda: `16UL << TOYOTA_PARAM_OFFSET`
  ⇒ 数字の 16 が両方に在ることを見る。片方だけ変えると**黙って upstream リミットに落ちる**
  (エラーにならないので実走するまで気づけない)。
  """
  car_src = _read(VALUES_PY)
  m = re.search(r'RAISED_STEER_LIMITS\s*=\s*\((\d+)\s*<<\s*8\)', car_src)
  assert m, 'values.py に ToyotaSafetyFlags.RAISED_STEER_LIMITS が無い'

  panda_src = _read(SAFETY_H)
  p = re.search(r'TOYOTA_PARAM_RAISED_STEER_LIMITS\s*=\s*(\d+)UL\s*<<\s*TOYOTA_PARAM_OFFSET', panda_src)
  assert p, 'toyota.h に TOYOTA_PARAM_RAISED_STEER_LIMITS が無い'

  assert m.group(1) == p.group(1), \
    f"フラグのビットがズレている: car={m.group(1)} panda={p.group(1)} = リミットが切り替わらない"


def test_panda_actually_switches_limits_on_the_flag():
  """フラグを定義しただけで tx_hook が使っていない、という落ち方を防ぐ。"""
  src = _read(SAFETY_H)
  assert 'toyota_raised_steer_limits ? TOYOTA_GS_F_TORQUE_STEERING_LIMITS' in src, \
    'tx_hook が toyota_raised_steer_limits で分岐していない = GS リミットが使われない'
  assert re.search(r'toyota_raised_steer_limits\s*=\s*GET_FLAG', src), \
    'toyota_init がフラグを読んでいない = 常に false になる'


def test_interface_sets_the_raised_limits_flag():
  """car 側がフラグを立てていなければ panda は upstream リミットのまま。"""
  src = _read(INTERFACE_PY)
  assert re.search(r'safetyParam\s*\|=\s*ToyotaSafetyFlags\.RAISED_STEER_LIMITS', src), \
    'interface.py が RAISED_STEER_LIMITS を safetyParam に立てていない'


def test_dsu_cruise_rx_check_for_mads_survives():
  """⚠ DSU_CRUISE (0x365) を RX に通していないと acc_main_on が更新されず、MADS の横が engage できない。"""
  src = _read(SAFETY_H)
  assert 'toyota_lka_unsupported_dsu_rx_checks' in src and 'TOYOTA_DSU_CRUISE_ADDR_CHECK' in src, \
    'unsupported-DSU 用の DSU_CRUISE RX check が落ちている = MADS の横が engage しなくなる'


def test_sdsu_engage_cancel_suppression_survives():
  """SDSU 構成で engage 時に cancel を送らないための抑止フレーム数。"""
  assert _literal(CARCONTROLLER_PY, 'SDSU_ENGAGE_CANCEL_SUPPRESS_TX') == 2


# ===========================================================================
# 2) 縦制御
# ===========================================================================

def test_pln3_auto_resume_survives():
  """PLN-3: 先頭停止からの自動発進。⚠ big model 限定なのが仕様 (small では None = 無効)。"""
  assert _literal(LONGCONTROL_PY, 'PLN3_GO_SUSTAIN_LEAD') == 0.5
  assert _literal(LONGCONTROL_PY, 'PLN3_LEAD_SUSTAIN') == 1.0
  assert _literal(LONGCONTROL_PY, 'PLN3_GO_SUSTAIN_NOLEAD') is None, \
    'small model で自動発進が有効になっている。big 限定が仕様'
  assert _literal(LONGCONTROL_PY, 'PLN3_GO_SUSTAIN_NOLEAD_BIG') == 1.5, \
    '青発進 sustain が 1.5s から変わっている (08-26 に 2.0 -> 1.5)'


def test_pln1_5_stop_threshold_survives():
  """PLN-1_5: chestnut 移行で CP.vEgoStopping が読まれなくなったので should_stop に直書きした値。

  ⚠ ここが 0.3 に戻ると停止が手前で緩み、**停止線を越えてから止まる**ようになる。
  """
  src = _read(DRIVE_HELPERS_PY)
  m = re.search(r'def should_stop\(.*?\n(.*?)(?=\ndef |\Z)', src, re.DOTALL)
  assert m, 'drive_helpers.py に should_stop が無い'
  body = m.group(1)
  assert re.search(r'\b0\.4\b', body), 'should_stop の停止閾値 0.4 が落ちている (PLN-1_5)'
  assert 'PLN-1_5' in src, 'PLN-1_5 の由来コメントが消えている (なぜ 0.4 なのかが失われる)'


def test_stop_distance_survives():
  """PLN-1_4: 停止線までの余裕。upstream 既定より手前で止める。"""
  assert _literal(LONG_MPC_PY, 'STOP_DISTANCE') == 8.5


def test_max_vel_err_survives():
  assert _literal(DRIVE_HELPERS_PY, 'MAX_VEL_ERR') == 5.0


def test_scc_vision_curve_tuning_survives():
  """SCC-V (PLN-1): カーブ手前の減速。実走で決めた曲率テーブル。"""
  assert _literal(SCC_VISION_PY, '_A_LAT_REG_MAX_BP') == [1.8, 2.4, 3.2]
  assert _literal(SCC_VISION_PY, '_A_LAT_REG_MAX_V') == [3.2, 3.2, 2.6]
  assert _literal(SCC_VISION_PY, '_ENTERING_SMOOTH_DECEL_V') == [-0.4, -1.2]
  assert _literal(SCC_VISION_PY, '_ENTERING_SMOOTH_DECEL_BP') == [1.1, 2.5]


def test_scc_map_gating_survives():
  """SCC-M: GPS が古いときに地図由来の減速を使わないためのゲート。"""
  assert _literal(SCC_MAP_PY, 'GPS_STALE_S') == 2.0
  src = _read(SCC_MAP_PY)
  assert 'V_TARGET_REJECT' in src, 'V_TARGET_REJECT が落ちている'


# ===========================================================================
# 3) 横制御
# ===========================================================================

def test_headroom_stage9_kp_table_survives():
  """Stage 9: 速度別 Kp。⚠ 実走で詰めたテーブルなので、長さと値の両方を守る。"""
  speeds = _literal(TORQUE_EXT_BASE_PY, 'KP_TQ_SPEEDS')
  kp = _literal(TORQUE_EXT_BASE_PY, 'KP_TQ_INTERP')
  assert speeds == [1, 1.5, 2.0, 3.0, 5, 7.5, 10, 15, 30]
  assert kp == [2.01, 2.33, 2.41, 2.83, 3.54, 3.69, 3.21, 1.97, 0.80]
  assert len(speeds) == len(kp), 'Kp テーブルの長さが噛み合っていない = interp が壊れる'


def test_nnlc_settled_trim_survives():
  """NNLC の落ち着き判定 + FF trim。"""
  assert _literal(NNLC_PY, 'SETTLED_LA_MIN') == 0.5
  assert _literal(NNLC_PY, 'SETTLED_RAMP_X') == [1.0, 2.5]
  assert _literal(NNLC_PY, 'SETTLED_RAMP_Y') == [0.0, 1.0]
  trim_la = _literal(NNLC_PY, 'SETTLED_TRIM_LA')
  trim_v = _literal(NNLC_PY, 'SETTLED_TRIM_V')
  tbl = _literal(NNLC_PY, 'SETTLED_TRIM_TBL')
  assert len(tbl) == len(trim_v), 'trim テーブルの行数が速度ブレークポイントと合っていない'
  assert all(len(row) == len(trim_la) for row in tbl), 'trim テーブルの列数が横 G ブレークポイントと合っていない'


def test_gs_f_nnlc_model_is_present():
  """GS 450h 用の NNLC モデル。⚠ 無いと NNLC が車種既定にフォールバックする。"""
  path = REPO_ROOT / 'openpilot/sunnypilot/neural_network_data/neural_network_lateral_control/LEXUS_GS_F.json'
  assert path.exists(), 'LEXUS_GS_F.json が無い = NNLC が GS 用モデルを使えない'
  assert path.stat().st_size > 0


def test_lane_centering_tuning_survives():
  """Lane Centering。⚠ 定数は全部 StarPilot 既定のまま = 3 つ変えたが実測で全部戻した。

  挙動そのものは test_lane_centering.py が見ている。ここは**載り漏れ検出**だけ。
  """
  assert _literal(LC_PY, '_MAX_OFFSET') == 0.3
  assert _literal(LC_PY, '_MAX_RAW_CORRECTION') == 0.004
  assert _literal(LC_PY, '_MAX_GAIN') == 0.30
  assert _literal(LC_PY, '_SMOOTH_TAU') == 0.4
  assert _literal(LC_PY, '_MIN_LANE_WIDTH') == 2.6
  assert _literal(LC_PY, '_MAX_LANE_WIDTH') == 4.8
  assert _literal(LC_PY, '_WIDTH_JUMP_LIMIT') == 0.50, '幅急変ガード (GS 独自の追加) が落ちている'


# ===========================================================================
# 4) modeld (chestnut / eGPU)
# ===========================================================================

def test_egpu_load_resilience_survives():
  """chestnut の cold load は実測 80.3s。upstream 既定の 60s では必ず timeout する。"""
  assert _literal(MODELD_PY, 'BIG_MODEL_TIMEOUT') == 150
  assert _literal(MODELD_PY, 'EGPU_LOAD_ATTEMPTS') == 5
  assert _literal(MODELD_PY, 'EGPU_LOCK_RETRY_WAIT') == 3.0


def test_long_smooth_cap_survives():
  """big model が積む long='.3' の頭打ち。⚠ 無いと縦が「ふんわり」になる。"""
  assert _literal(MODELD_PY, 'LONG_SMOOTH_SECONDS_MAX') == 0.15


def test_long_smooth_cap_is_actually_applied():
  """定数が在るだけでは足りない。**実際に効かせている**ことまで見る。

  ⚠ 追従で定数だけ cherry-pick され、使用箇所が upstream 版に戻る落ち方をする。
  """
  assert re.search(r'min\s*\(.*LONG_SMOOTH_SECONDS_MAX', _read(MODELD_PY)), \
    'LONG_SMOOTH_SECONDS_MAX が min() で使われていない = 定数はあるが効いていない'


# ===========================================================================
# 5) UI (mici)
# ===========================================================================

def test_model_selector_shows_every_folder():
  """⚠ ホワイトリスト方式に戻ると 77 本中 59 本が画面から消える (2026 Deep RL Models が丸ごと落ちる)。

  chestnut 接続時の TT / IDM / HBM が選べなくなるので、eGPU 運用が成立しなくなる。
  """
  src = _read(MICI_MODELS_PY)
  assert '"release models"' not in src and "'release models'" not in src, \
    'models 画面がフォルダのホワイトリストに戻っている = 大半のモデルが選べない'


def test_lane_centering_ui_is_wired_into_settings():
  """設定画面の DEC の右に Lane Centering が並んでいること。"""
  src = _read(MICI_TOGGLES_PY)
  assert 'lane_centering' in src, 'mici の設定に Lane Centering が載っていない'
  assert 'DynamicExperimentalControl' in src, 'DEC トグルが落ちている'


# ===========================================================================
# 6) Lane Centering の保存先 (params から切り離した設計)
# ===========================================================================

LC_KEYS = ['LaneCentering', 'LaneCenterOffset', 'LaneCenteringE2EAuthority', 'LaneCenteringPauseOnSignal']


def test_lane_centering_keys_are_not_declared_in_params_keys_h():
  """⚠ ここに載せると `.so` を焼き直さない限り実機で UnknownKeyName になる。

  焼く運用は 2026-08-27 に捨てた。⇒ 二度と載せない。
  """
  src = _read(PARAMS_KEYS_H)
  leaked = [k for k in LC_KEYS if k in src]
  assert not leaked, f"params_keys.h に {leaked} が戻っている。焼かないと実機で読めなくなる"


def test_lane_centering_save_location_is_outside_openpilot_params():
  """⚠ `/data/params/` の下に戻すと `clearAll` が毎ブート消す (manager が起動時に 4 回呼ぶ)。"""
  m = re.search(r"^PARAM_DIR\s*=\s*['\"]([^'\"]+)['\"]", _read(LC_PARAMS_PY), re.MULTILINE)
  assert m, 'lane_centering_params.py に PARAM_DIR が無い'
  assert not m.group(1).startswith('/data/params/'), \
    f"PARAM_DIR が {m.group(1)} = openpilot の params 配下に戻っている。clearAll に毎ブート消される"


def test_lane_centering_read_does_not_take_a_params_object():
  """⚠ `.so` を焼いた端末では Params.get(..., return_default=True) が**必ず既定値を返す**。

  Params を先に見る実装だと**ファイルの値が黙って無視される**。引数ごと受け取れないことを固定する。
  """
  for node in ast.parse(_read(LC_PARAMS_PY)).body:
    if isinstance(node, ast.FunctionDef) and node.name == 'read':
      names = [a.arg for a in node.args.args]
      assert names == ['key'], f"read() が {names} を取る。Params を渡せると焼いた端末で壊れる"
      return
  pytest.fail('lane_centering_params.py に read() が無い')


# ===========================================================================
# 7) 上流 API の契約 (GS 側が依存しているもの)
# ===========================================================================

UPSTREAM_CONTRACT = [
  (
    'openpilot/sunnypilot/models/helpers.py',
    ['get_active_bundle', 'get_active_source', 'ACTIVE_BUNDLE_KEYS'],
    'chestnut 有無で active bundle のスロットを選ぶ。GS 側の restore_big_bundle を捨てた代替がこれ',
  ),
  (
    'openpilot/selfdrive/modeld/helpers.py',
    ['usbgpu_present'],
    'chestnut が挿さっているかの判定。modeld と UI の両方が使う',
  ),
]


@pytest.mark.parametrize(('rel', 'names', 'why'), UPSTREAM_CONTRACT)
def test_upstream_api_that_gs_depends_on_still_exists(rel, names, why):
  missing = [n for n in names if n not in _toplevel_names(rel)]
  assert not missing, f"{rel} から {missing} が消えた。GS 改造の前提が崩れている ({why})"


# ⚠ 2026-08-27 に上流のスロット分離へ寄せて捨てた GS 独自関数。
#   定義だけ消して呼び出し元が残ると、実車で modeld が起動時に落ちる。
REMOVED_GS_HELPERS = ['active_bundle_is_big', 'restore_big_bundle']


@pytest.mark.parametrize('name', REMOVED_GS_HELPERS)
def test_no_leftover_reference_to_removed_gs_helpers(name):
  hits = []
  for path in (REPO_ROOT / 'openpilot').rglob('*.py'):
    if '__pycache__' in path.parts or path.resolve() == Path(__file__).resolve():
      continue
    try:
      if name in path.read_text(encoding='utf-8'):
        hits.append(str(path.relative_to(REPO_ROOT)))
    except (OSError, UnicodeDecodeError):
      continue
  assert not hits, f"捨てたはずの {name} への参照が残っている: {hits}"


# ===========================================================================
# 8) 追従そのものの事故 (cherry-pick の取りこぼし)
# ===========================================================================

GS_TOUCHED_FILES = [
  VALUES_PY, INTERFACE_PY, CARCONTROLLER_PY,
  LONGCONTROL_PY, DRIVE_HELPERS_PY, LONG_MPC_PY,
  MODELD_PY, NNLC_PY, TORQUE_EXT_BASE_PY,
  SCC_VISION_PY, SCC_MAP_PY,
  LC_PY, LC_PARAMS_PY,
  MICI_MODELS_PY, MICI_TOGGLES_PY,
  'openpilot/selfdrive/ui/mici/widgets/lane_centering.py',
  'openpilot/selfdrive/controls/controlsd.py',
  'openpilot/selfdrive/controls/lib/latcontrol_torque.py',
  'openpilot/sunnypilot/selfdrive/controls/lib/dec/dec.py',
  'openpilot/sunnypilot/selfdrive/controls/lib/latcontrol_torque_ext.py',
]


@pytest.mark.parametrize('rel', GS_TOUCHED_FILES)
def test_gs_touched_files_parse_and_have_no_conflict_markers(rel):
  src = _read(rel)
  for marker in ('<<<<<<<', '>>>>>>>'):
    assert marker not in src, f"{rel} に merge conflict marker `{marker}` が残っている"
  if rel.endswith('.py'):
    try:
      ast.parse(src, filename=rel)
    except SyntaxError as e:
      pytest.fail(f"{rel} が構文エラー: line {e.lineno}: {e.msg}")


def test_pln1_6_creep_fade_keeps_braking_compensation():
  """PLN-1_6: creep 帯フェードは「ゼロ付近の要求を保持したとき」だけに効かせる。

  07-15 のフェードは v<0.5 で PCM 補償を丸ごと落とすので、停止接近 (要求 -0.45) でも
  補償が消え、実行率が 47-69% (他帯は 77-85%) まで落ちて最後の ~1.3m をじりじり進んでいた。
  ⚠ この門が消えると停止がまた緩む。⚠ フェード自体を消すと 07-15 の前後脈動が戻る
  (発火条件は 08-28 時点でも 00f に 1 件残っている)。**両方必要**。
  """
  src = _read(CARCONTROLLER_PY)
  assert 'PLN-1_6' in src, 'PLN-1_6 の由来コメントが消えている (なぜ門があるかが失われる)'
  m = re.search(r'elif self\.CP\.carFingerprint == CAR\.LEXUS_GS_F:(.*?)\n        else:',
                src, re.DOTALL)
  assert m, 'carcontroller.py の GS_F creep フェード分岐が見つからない'
  body = m.group(1)
  assert '_comp_frac' in body, 'creep 帯フェードが落ちている (07-15 の脈動対策)'
  assert re.search(r'actuators\.accel\s*<\s*-0\.2', body), \
      'PLN-1_6 の「明確に減速中なら補償を残す」門が落ちている'


def test_dec_radar_mode_checks_slow_down_before_lead():
  """DEC の `_radar_mode` は slow_down を lead より先に見ること (GS 450h 改造)。

  上流の順 (lead が先) に戻ると、前車を検知した瞬間に e2e の視覚先読み減速が
  planner の候補から丸ごと外れ、「set 速度のまま車列に近づいて急ブレーキ」が再発する。
  route 015 の実測 (前車ありの減速シーン 51 件、`archive/probes/_dec_mode_shadow.py`):
    上流順 = blended  0.2% (16/7992)
    この順 = blended 65.4% (5227/7992)
  停止時も、上流順は停止**直前**が blended 0.0% で停止後に 82.6% へ切り替わるため、
  その切替の隙に MPC がクリープを足して前車へ詰める (user 実体験)。
  """
  src = _read('openpilot/sunnypilot/selfdrive/controls/lib/dec/dec.py')
  body = src.split('def _radar_mode')[1].split('def update')[0]
  i_dep = body.index('_lead_departing')
  i_slow = body.index('self._has_slow_down')
  i_lead = body.index('self._has_lead_filtered')
  assert i_dep < i_slow, 'launch fix v3 が slow_down より後ろに落ちた = 発進のもたつきが再発する'
  assert i_slow < i_lead, 'slow_down が lead より後ろ = 上流順に戻っている (先読み減速が消える)'


def test_gs_touched_file_list_is_not_stale():
  """⚠ このリスト自体が腐るのを防ぐ。ファイルが移動/改名されたら _read が fail する。"""
  missing = [rel for rel in GS_TOUCHED_FILES if not (REPO_ROOT / rel).exists()]
  assert not missing, f"GS 改造ファイルが見つからない (上流が移動/改名した?): {missing}"
