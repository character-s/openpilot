"""Lane Centering の設定 4 つの定義と読み書き。

⚠⚠ **なぜ openpilot の Params (`/data/params`) を使わないのか** (2026-08-26 に c4 実機で確定):

`common/params.cc` の `clearAll` は **ホワイトリストに無いファイルを無条件に unlink** する:

    auto it = keys.find(de->d_name);
    if (it == keys.end() || (it->second.flags & key_flag)) {
      unlink(getParamPath(de->d_name).c_str());
    }

`keys` の実体は **`libparams_c.so` にコンパイルされ込まれた表**で、`system/manager/manager.py` が
manager 起動のたびに `clear_all` を 4 回呼ぶ。sunnypilot の release 配布はビルド定義
(SConstruct / SConscript) を落としているため c4 では scons が走らず、`common/params_keys.h` に
キーを足しても **`.so` は古いまま = ホワイトリストに載らない**。⇒ `/data/params/d` に置いた値は
**manager が起動するたびに消える**。c4 は ACC 連動で毎回コールドブートするので、
**エンジンをかけるたびに設定が既定値に戻る** = 実用にならない。

⇒ **この 4 つは `/data/params` の外に置く**。`clearAll` は `/data/params/<prefix>/` の中しか
走査しないので、`/data/params_fork/` は**構造的に対象外**になる。`.so` を焼いて配る必要が消え、
配布バイナリの世代と fork の Python の噛み合わせを人間が保証する運用も要らなくなる
(`params.py` は `.so` から ctypes で 16 シンボルを引いており、古い `.so` と新しい `params.py` が
噛み合わないと import 時に params が全滅 = manager 起動不能になりうる)。

`/data/params_fork` という名前は `/data/params` の**隣に並べて関係を示す**ため。中の構造も
params と同じ (`d/` に値、その親に `.tmp_value_*`) にしてあるので、将来 params 本体へ
戻すときは `d/` の中身をそのまま移せる。fork が今後足す設定もここに置けばよい。

⚠ **代償**: params の `BACKUP` フラグに乗らないので端末初期化時の自動復元が効かない。4 項目を
UI で入れ直すことになる。それと引き換えに「起動のたびに消える」を消している。

⚠ **Params は読みも書きも一切経由しない**。経由すると `.so` が焼かれた端末とそうでない端末で
挙動が変わり、今回の問題が別の形で再発する (焼かれた端末では `Params.get(..., return_default=True)`
が**必ず既定値を返す**ので、Params を先に見るとファイルの値が黙って無視される)。値の出所は
このモジュールが読むファイル 1 本に固定する。

このモジュールが 4 つの定義 (既定値・範囲・UI の選択肢) の**唯一の出所**。制御側
(`lane_centering.py`) と UI 側 (`selfdrive/ui/mici/widgets/lane_centering.py`) の両方がここを見る。

⚠ 依存は標準ライブラリだけに保つこと。`lane_centering.py` が numpy 以外を引かないのと同じ理由で、
cereal (→ opendbc submodule) を経由しないと import できない状態にすると PC 側で単体テストが
回せなくなる。
"""
import os
import tempfile

# 値の実体。⚠⚠ **`/data/params` の外であることが要**。中に置くと manager 起動のたびに
# `clearAll` に消される (モジュール docstring 参照)。
# ⚠ `PARAM_DIR` はテストが monkeypatch する差し替え点なので、参照側は
# `from ... import PARAM_DIR` ではなく **モジュール属性で** 引くこと。
PARAM_DIR = '/data/params_fork/d'

KEY_ENABLED = "LaneCentering"
KEY_OFFSET = "LaneCenterOffset"
KEY_AUTHORITY = "LaneCenteringE2EAuthority"
KEY_PAUSE_ON_SIGNAL = "LaneCenteringPauseOnSignal"

# ⚠ 「値が読めない」= 既定値であって「前回値を維持」ではない — 前回値だと「LaneCentering だけ
# ファイルを置いた」ときに他の 3 つが更新されず、項目ごとに世代の違う値が混ざる。
DEFAULTS = {
  KEY_ENABLED: False,
  KEY_OFFSET: 0.0,
  KEY_AUTHORITY: 1.0,
  KEY_PAUSE_ON_SIGNAL: True,
}

# UI が回す選択肢。⚠ offset の符号はデバイス座標 (y は左が正) に従い **+ が左寄せ**。
# CameraOffset の説明文 "Left(+ values) or Right (- values)" と同じ向き。
# 0.1m 刻みなのは、実測の平衡点移動が 0.40 → 0.22m = 18cm なのに対し、それより細かく刻んでも
# 体感で区別できないため。上下限は制御側の `_MAX_OFFSET` (0.3) と一致させる。
OFFSET_CHOICES = (-0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3)
# authority = モデルが「自信を持って大きく外している」ときに補正を譲る強さ。1.0 = 全部譲る (既定)。
# 0.25 刻み。⚠ 0 にすると構造変化の保護が消える (08-26 実測: 幅 3.0→4.0m で la +0.234 m/s²) ので、
# 0 を選べること自体は残しつつ既定は 1.0 のまま。
AUTHORITY_CHOICES = (0.0, 0.25, 0.5, 0.75, 1.0)


def offset_label(v: float) -> str:
  """offset の表示。⚠ 符号はデバイス座標 (y は左が正) なので **+ が左**。

  数値だけだと「+ はどっち」が毎回分からなくなるので left/right を併記する。
  ⚠ UI (`BigMultiToggle`) は表示ラベルを index() で引くので、選択肢の中で**重複してはいけない**。
  """
  if abs(v) < 0.005:
    return "centered"
  return f"{abs(v):.2f} m {'left' if v > 0 else 'right'}"


def authority_label(v: float) -> str:
  return f"{int(round(v * 100))}%"


def read(key):
  """値のファイルを読む (bytes)。無ければ (= まだ一度も書かれていなければ) None。

  ⚠ **Params は経由しない — 引数として受け取りもしない** (理由 = モジュール docstring)。
  """
  try:
    with open(os.path.join(PARAM_DIR, key), 'rb') as f:
      return f.read()
  except OSError:
    return None


def read_bool(key):
  default = DEFAULTS[key]
  raw = read(key)
  # C++ の get_bool と同じ「'0' 以外は真」
  return default if raw is None else raw.strip() not in (b'', b'0')


def read_float(key):
  default = DEFAULTS[key]
  raw = read(key)
  if raw is None:
    return default
  try:
    return float(raw)   # float() は bytes を受け付け前後の空白も無視する
  except ValueError:
    return default


def _write_file(key, text: str) -> bool:
  """`common/params.cc` の putParam と同じ手順 (tmp に書く → fsync → rename → 親 dir を fsync) で差し替える。

  ⚠ 手順を守るのは **controlsd が 1Hz でこのファイルを読んでいる**ため。普通に open('w') すると
  truncate と write の間に読まれて「空 = 既定値」に落ちる瞬間がある。rename は atomic なので lock は要らない
  (書き手は UI 1 本、controlsd は読むだけ)。
  ⚠ tmp は `d/` の**外** (= `PARAM_DIR` の親) に作る。`d/` 内に作ると列挙にゴミが混じる。
  """
  key_path = os.path.join(PARAM_DIR, key)
  parent = os.path.dirname(PARAM_DIR) or '.'
  # ⚠ makedirs も mkstemp も try の中で。書けない端末があり (PC 側の replay など)、外に出すと
  #   OSError が UI まで抜けて設定画面が落ちる。
  tmp_fd, tmp_path = None, None
  try:
    # ⚠ `/data/params` と違って openpilot の誰も作ってくれないので、最初の書き込みで作る。
    #   読み側は「無ければ既定値」で動くので、作るのは書きのときだけでよい。
    os.makedirs(PARAM_DIR, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix='.tmp_value_', dir=parent)
    os.write(tmp_fd, text.encode('utf-8'))
    os.fsync(tmp_fd)
    os.close(tmp_fd)
    tmp_fd = None

    os.replace(tmp_path, key_path)
    tmp_path = None

    try:
      dir_fd = os.open(PARAM_DIR, os.O_RDONLY)
      try:
        os.fsync(dir_fd)
      finally:
        os.close(dir_fd)
    except OSError:
      pass    # dir の fsync に失敗しても rename 自体は済んでいる
    return True
  except OSError:
    return False
  finally:
    if tmp_fd is not None:
      os.close(tmp_fd)
    if tmp_path is not None:
      try:
        os.unlink(tmp_path)
      except OSError:
        pass


def write(key, value) -> bool:
  """`PARAM_DIR/<key>` へ tmp → fsync → rename で同期的に書く。成功したら True (UI はこれで表示を戻す)。

  ⚠ Params は経由しない (理由 = モジュール docstring)。put() は block=False 既定で直後の読みが古い値を
    返しうる (block=True でも 1 つズレる)。
  ⚠ bool/float は値の型でなくキーの型 (DEFAULTS) で決める — write(KEY_ENABLED, 0) が "0.0" になると読み戻しが True。
  ⚠ float は repr() (":.2f" は選択肢を細かくしたとき黙って丸まる)。
  """
  if isinstance(DEFAULTS.get(key), bool):
    text = '1' if value else '0'
  else:
    text = repr(float(value))
  return _write_file(key, text)
