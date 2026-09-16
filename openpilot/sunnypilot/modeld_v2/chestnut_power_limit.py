"""chestnut (eGPU) の SMU パッケージ電力上限を絞る (GS450h 追加)。

⚠⚠ **なぜ絞るのか (09-16 に決定)**:
GPU の定常消費は **48-50W しかない**のに、XT60 入力の電流は瞬間 **5.9A = 19.65V 換算で 116W** に届く
(route `073` を `_chestnut_power.py` で実測)。これはブースト時のトランジェントで、
**PD 充電器の定格 (典型 20V/5A = 100W) を超えている**。充電器が過電流保護で一瞬落ちると
USB リンクが切れ、`Device hang detected` になる。
⚠ トランジェントは ms オーダーで、`chestnutState` は 10Hz ⇒ **ログにはほとんど写らない**。
「ハング直前の電圧が正常だった」は給電説の反証にならない。

**上流 #2023** (`PeterPhuTran`、同型の comma four + chestnut) の実測:
「無制限 = engage 後 46 秒で毎回 hang / **60W cap = 88 分の通勤で 0 件・4/4 走行クリーン** /
CTMV2 を 60W で回して 36ms median・42ms max・frame drop 0 = **cap にスループットの代償は無い**」。
⚠ **stock の上限は我々の実機で 170W** (#2023 著者の Radeon 9060 16GB は 160W) =
chestnut の推論設計点 100W を大きく超えた値が既定になっている。**絞るのは異常値の是正**であって
性能を削る操作ではない。

⚠ **値の置き場は `/data/params_fork/d` で openpilot の Params ではない**。理由は
`lane_centering_params.py` の docstring と同じ = `common/params.cc` の `clearAll` が
ホワイトリストに無いキーを manager 起動のたびに消すため (`.so` を焼き直さない限り載らない)。
⇒ **c4 上でファイルを置くだけで値を変えられる** = 再配信なしで 80 → 60 を試せる。

⚠ **SMU 呼び出しが失敗しても load は止めない**。上流 #2023 は「絞れない給電に無制限で流すよりは
落とす方がまし」として load 失敗にするが、GS 450h は **big を粘る方針** (small との実力差が大きい
= `load_big` のコメント) なので、警告だけ出して続行する。絞れたかどうかは
**`chestnutState.powerLimitW` が publish されている**ので走行後に検証できる。

⚠ **`tinygrad.device` の import は関数の中に置くこと**。モジュールスコープで tinygrad に触ると、
issue #1964 で判明した「import 時に AMD デバイスを開いて flock を握ったまま失敗する」を
自分で再現しかねない (上流の修正は #2033)。

⚠ **`swaglog` も遅延 import で握る**。PC には zmq が無く `openpilot.common.swaglog` が import できない
ため、モジュールスコープで引くと**単体テストが PC で回せなくなる** (`lane_centering_params.py` の
「依存は標準ライブラリだけに保つ」と同じ理由)。⇒ **モジュールスコープの依存は `os` だけ**。
"""
import os

# ⚠ `/data/params` の外であることが要 (モジュール docstring 参照)。
# ⚠ テストが monkeypatch する差し替え点なので、参照側は `from ... import PARAM_DIR` ではなく
#    **モジュール属性で** 引くこと (`lane_centering_params.py` と同じ約束)。
PARAM_DIR = '/data/params_fork/d'

KEY_POWER_LIMIT = "ChestnutPowerLimit"

# chestnut は 100W の推論を設計点にしている ⇒ それより上は受け付けない (#2023 のレビューでも
# 「120-150 は不要」と指摘され、上流も 40-100W に狭めた)。40W 未満は推論そのものが痩せる
# 懸念があるので下限に置く。
POWER_LIMIT_MIN_W = 40
POWER_LIMIT_MAX_W = 100

# **既定 = 80W (user 09-16 決定)**。GPU の定常が 48-50W なので 80W は定常に 6 割の余裕を残しつつ、
# 実測ピーク 116W のトランジェントは確実に削る。
# ⏭ 効果が足りなければ **60W** (#2023 著者が 88 分 + 4/4 クリーンを出した実績値) へ下げる。
# ⚠ **0 を書けば stock に戻る** = 性能が落ちたときの退避先。
DEFAULT_LIMIT_W = 80


def get_power_limit() -> int:
  """要求する上限 [W]。**0 = stock (絞らない)**。

  ⚠ ファイルが無ければ既定値 (= 絞る)。「読めない = 前回値」にはしない
  (`lane_centering_params.py` と同じ理由 — 項目ごとに世代の違う値が混ざるのを避ける)。
  """
  try:
    with open(os.path.join(PARAM_DIR, KEY_POWER_LIMIT), 'rb') as f:
      raw = f.read()
  except OSError:
    return DEFAULT_LIMIT_W

  try:
    limit_w = int(float(raw))   # float() は bytes を受け付け前後の空白も無視する
  except ValueError:
    return DEFAULT_LIMIT_W

  if limit_w <= 0:
    return 0
  return max(POWER_LIMIT_MIN_W, min(POWER_LIMIT_MAX_W, limit_w))


def _log(**kwargs) -> None:
  """cloudlog へ 1 行残す。⚠ 遅延 import + 失敗を握る (モジュール docstring 参照)。

  ログが出せないこと自体でロードを止めない。
  """
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.event("chestnut power limit", **kwargs)
  except Exception:
    pass


def apply_power_limit(limit_w: int) -> int | None:
  """開いている AMD デバイスの SMU に上限を設定し、**読み戻した値**を返す。0 なら何もしない。

  読み戻すのは「送れた」と「効いた」が別だから。要求と違う値が返ったら `error=True` で残す。
  ⚠ 例外は投げない (モジュール docstring) — 呼び出し元はそのままロードへ進む。
  """
  if limit_w <= 0:
    return None

  try:
    from tinygrad.device import Device   # ⚠ 関数内 import (モジュール docstring 参照)
    smu = Device["AMD"].iface.dev_impl.smu
    smu._send_msg(smu.smu_mod.PPSMC_MSG_SetPptLimit, limit_w, timeout=100)
    applied = int(smu._send_msg(smu.smu_mod.PPSMC_MSG_GetPptLimit, 0, read_back_arg=True, timeout=100))
    _log(requested=limit_w, applied=applied, error=applied != limit_w)
    return applied
  except Exception as e:
    # ⚠ 型とメッセージは残す。ここは 3 行しかないので traceback は要らない。
    _log(requested=limit_w, applied=None, error=True, exc=repr(e))
    return None
