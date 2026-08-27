"""lane_centering_params のテスト。

ここが 4 キーの定義の唯一の出所で、制御 (`lane_centering.py`) と UI
(`selfdrive/ui/mici/widgets/lane_centering.py`) の両方がここを見るので、
**両者が食い違わないこと**を機械で押さえる。

⚠ UI ウィジェット本体は pyray を引くので PC では import できない。だから UI が依存する
ロジック (表示ラベル・選択肢) は全部こちら側に置いてある。
"""
import os

import pytest

from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering_params as lcp
from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering as lc


@pytest.fixture
def param_dir(tmp_path, monkeypatch):
  d = tmp_path / "d"
  d.mkdir()
  monkeypatch.setattr(lcp, 'PARAM_DIR', str(d))
  return d


# ── 保存先 (ここが 2026-08-26 に踏んだ問題の本体) ──────────────────────

def test_the_save_location_is_outside_openpilot_params():
  """⚠⚠ 保存先が `/data/params` の中に**無い**こと。

  `common/params.cc` の `clearAll` は **ホワイトリストに無いファイルを無条件に unlink** し、
  `system/manager/manager.py` が manager 起動のたびにそれを 4 回呼ぶ。ホワイトリストの実体は
  `libparams_c.so` に焼かれた表で、sunnypilot の release 配布はビルド定義 (SConstruct /
  SConscript) を落としているため、fork が `common/params_keys.h` にキーを足しても c4 では
  **`.so` が古いまま = 載らない**。⇒ `/data/params/d` に置くと **manager が起動するたびに消える**。
  c4 は ACC 連動で毎回コールドブートするので、エンジンをかけるたびに設定が既定値へ戻っていた。

  ⚠ ここを `/data/params/...` に戻すと同じ症状が再発する。機械で固定しておく。
  """
  assert not lcp.PARAM_DIR.startswith('/data/params/')
  assert lcp.PARAM_DIR.startswith('/data/')


def test_temp_and_lock_stay_inside_the_save_location():
  """tmp と `.lock` の置き場所が保存先の中に収まっていること。

  ⚠ `_write_file` は tmp を `os.path.dirname(PARAM_DIR)` に作る (params.cc と同じ作法で、
  `d/` の中に作ると列挙にゴミが混じるため)。**保存先を動かすと tmp の置き場所も一緒に動く**ので、
  `/data` 直下のような共有の場所へ飛び出していないことを確かめる。
  """
  parent = os.path.dirname(lcp.PARAM_DIR)
  assert parent not in ('', '/', '/data'), parent
  assert lcp.PARAM_DIR.startswith(parent + '/')


def test_params_keys_h_no_longer_declares_the_keys():
  """`common/params_keys.h` に 4 つの宣言が**残っていない**こと。

  ⚠ 残すと「openpilot の Params にも同じ名前の設定がある」状態になる。`.so` が焼かれた端末では
  `Params.get(..., return_default=True)` が既定値を返すので、二重管理の温床になるだけで
  実利が無い。保存先を params の外に出した以上、宣言も消しておく (upstream 差分も減る)。
  """
  here = os.path.dirname(__file__)
  keys_h = os.path.abspath(os.path.join(here, '..', '..', '..', '..', '..', 'common', 'params_keys.h'))
  src = open(keys_h, encoding='utf-8').read()
  for key in lcp.DEFAULTS:
    assert f'"{key}"' not in src, f"{key} が params_keys.h に残っている"


# ── 定義の整合 (ここがずれると画面と制御が別の値を指す) ──

def test_defaults_cover_every_key_with_the_right_type():
  """DEFAULTS が 4 つとも埋まっていて、型が bool / float で揃っていること。

  ⚠ 欠けると `read_bool` / `read_float` が KeyError で落ちる (= controlsd の 1Hz ループが
  例外を握って前回値のまま走り続ける = 設定が効かない理由が見えなくなる)。
  ⚠ 型がずれると `write()` の書式判定 (キーの型で bool / float を決める) が狂う。
  """
  assert set(lcp.DEFAULTS) == {lcp.KEY_ENABLED, lcp.KEY_OFFSET,
                               lcp.KEY_AUTHORITY, lcp.KEY_PAUSE_ON_SIGNAL}
  assert isinstance(lcp.DEFAULTS[lcp.KEY_ENABLED], bool)
  assert isinstance(lcp.DEFAULTS[lcp.KEY_PAUSE_ON_SIGNAL], bool)
  assert isinstance(lcp.DEFAULTS[lcp.KEY_OFFSET], float)
  assert isinstance(lcp.DEFAULTS[lcp.KEY_AUTHORITY], float)


def test_offset_choices_stay_inside_the_controller_clip():
  """UI が出す選択肢が制御側のクリップ幅を超えないこと (超えると選んでも効かない)。"""
  assert max(abs(v) for v in lcp.OFFSET_CHOICES) <= lc._MAX_OFFSET
  assert 0.0 in lcp.OFFSET_CHOICES              # 既定値が選択肢に含まれること
  assert lcp.DEFAULTS[lcp.KEY_OFFSET] in lcp.OFFSET_CHOICES


def test_authority_choices_cover_the_default_and_the_unit_range():
  assert min(lcp.AUTHORITY_CHOICES) >= 0.0 and max(lcp.AUTHORITY_CHOICES) <= 1.0
  assert lcp.DEFAULTS[lcp.KEY_AUTHORITY] in lcp.AUTHORITY_CHOICES


@pytest.mark.parametrize("choices,label_fn", [(lcp.OFFSET_CHOICES, lcp.offset_label),
                                              (lcp.AUTHORITY_CHOICES, lcp.authority_label)])
def test_labels_are_unique(choices, label_fn):
  """UI は表示ラベルを index() で引くので、重複すると別の選択肢に飛ぶ。"""
  labels = [label_fn(v) for v in choices]
  assert len(set(labels)) == len(labels)


def test_offset_label_sign_follows_device_frame():
  """+ が左。⚠ ここを逆にすると画面と実車の動きが反対になる。"""
  assert "left" in lcp.offset_label(0.2)
  assert "right" in lcp.offset_label(-0.2)
  assert lcp.offset_label(0.0) == "centered"


# ── 読み書きの往復 ──

@pytest.mark.parametrize("value", [True, False])
def test_bool_round_trip_via_file(param_dir, value):
  assert lcp.write(lcp.KEY_ENABLED, value)
  assert lcp.read_bool(lcp.KEY_ENABLED) is value


@pytest.mark.parametrize("value", [-0.3, -0.1, 0.0, 0.25, 1.0])
def test_float_round_trip_via_file(param_dir, value):
  assert lcp.write(lcp.KEY_OFFSET, value)
  assert lcp.read_float(lcp.KEY_OFFSET) == pytest.approx(value)


@pytest.mark.parametrize("falsy", [False, 0, 0.0, "", b""])
def test_bool_key_uses_key_type_not_value_type(param_dir, falsy):
  """bool キーは**キーの型**で判定すること。値の型で分岐してはいけない。

  `isinstance(value, bool)` で分岐すると、bool でない偽値 (int の 0 など) が float 扱いになって
  `"0.00"` が書かれる。読み戻すと `as_bool(b"0.00")` は「'0' と等しくない」ので **True** になり、
  C++ の get_bool も同じ判定なので Params 経由でも同じ ⇒ OFF にしたつもりが ON になる。
  """
  assert lcp.write(lcp.KEY_ENABLED, falsy)
  assert (param_dir / lcp.KEY_ENABLED).read_bytes() == b"0"
  assert lcp.read_bool(lcp.KEY_ENABLED) is False


@pytest.mark.parametrize("truthy", [True, 1, 0.5])
def test_bool_key_accepts_non_bool_truthy(param_dir, truthy):
  assert lcp.write(lcp.KEY_ENABLED, truthy)
  assert (param_dir / lcp.KEY_ENABLED).read_bytes() == b"1"
  assert lcp.read_bool(lcp.KEY_ENABLED) is True


def test_write_never_goes_through_params(param_dir):
  """書きは必ずファイル経路であること (Params は受け取りもしない)。

  ⚠ Params の put() は既定が block=False で、直後にファイルが更新されている保証がない
  (2026-08-26 実機)。書けたと言った直後に controlsd が古い値を読む状態を作らないため、
  書きは _write_file 一本に固定してある。引数として Params を渡せないことで担保する。
  """
  with pytest.raises(TypeError):
    lcp.write(lcp.KEY_ENABLED, True, object())


def test_float_key_is_not_coerced_to_bool(param_dir):
  """逆向き: float キーに 0.0 を書いても bool の '0' にはしないこと。"""
  assert lcp.write(lcp.KEY_OFFSET, 0.0)
  assert (param_dir / lcp.KEY_OFFSET).read_bytes() == b"0.0"


def test_written_bool_is_readable_by_the_c_params_convention(param_dir):
  """C++ 側の get_bool は '0' 以外を真と見るので、書式もそれに揃える。"""
  lcp.write(lcp.KEY_ENABLED, True)
  assert (param_dir / lcp.KEY_ENABLED).read_bytes() == b"1"
  lcp.write(lcp.KEY_ENABLED, False)
  assert (param_dir / lcp.KEY_ENABLED).read_bytes() == b"0"


def test_read_takes_no_params_argument():
  """読みは Params を経由しない — 引数として渡せもしないこと。

  ⚠⚠ `.so` が焼かれた端末では `Params.get(key, return_default=True)` が**必ず**既定値を返す。
  Params を先に見る実装だとそこでファイルの値が黙って無視され、**端末によって挙動が変わる**。
  今回の「毎ブート消える」と同じ種類の再発なので、受け取れないことで担保する。
  """
  with pytest.raises(TypeError):
    lcp.read(lcp.KEY_OFFSET, object())
  with pytest.raises(TypeError):
    lcp.read_bool(lcp.KEY_ENABLED, object())
  with pytest.raises(TypeError):
    lcp.read_float(lcp.KEY_OFFSET, object())


def test_write_then_read_is_immediately_consistent(param_dir):
  """書いた直後に読むと必ず新しい値が返ること。

  ⚠ これが成り立つのは書きが同期 (fsync + rename) だから。Params の put() に任せると
  非同期キューに載って、直後の読みが古い値を返しうる。UI は「書けたら表示を更新」し、
  controlsd は 1Hz で読むので、ここがズレると画面と制御が食い違う。
  """
  for value in (-0.3, 0.0, 0.25):
    assert lcp.write(lcp.KEY_OFFSET, value)
    assert lcp.read_float(lcp.KEY_OFFSET) == pytest.approx(value)


def test_float_is_written_without_losing_precision(param_dir):
  """repr 書式なので選択肢を細かくしても丸まらないこと。

  ⚠ f"{v:.2f}" だと 0.125 が "0.12" になり、書いた値と読み戻した値が一致しなくなる。
  焼いた Params が書く表現 (0.25 -> "0.25", 0.0 -> "0.0") とも揃う。
  """
  for value in (0.125, -0.075, 0.3333):
    assert lcp.write(lcp.KEY_OFFSET, value)
    assert lcp.read_float(lcp.KEY_OFFSET) == pytest.approx(value)
  lcp.write(lcp.KEY_OFFSET, 0.0)
  assert (param_dir / lcp.KEY_OFFSET).read_bytes() == b"0.0"
  lcp.write(lcp.KEY_AUTHORITY, 0.25)
  assert (param_dir / lcp.KEY_AUTHORITY).read_bytes() == b"0.25"


def test_missing_key_falls_back_to_default(param_dir):
  assert lcp.read_bool(lcp.KEY_ENABLED) is lcp.DEFAULTS[lcp.KEY_ENABLED]
  assert lcp.read_float(lcp.KEY_AUTHORITY) == lcp.DEFAULTS[lcp.KEY_AUTHORITY]


def test_garbage_value_falls_back_to_default(param_dir):
  (param_dir / lcp.KEY_OFFSET).write_bytes(b"not-a-number")
  assert lcp.read_float(lcp.KEY_OFFSET) == lcp.DEFAULTS[lcp.KEY_OFFSET]


# ── 書き込みの作法 (走行中に controlsd が 1Hz で読んでいる) ──

def test_write_leaves_no_temp_files(param_dir):
  """tmp が params ディレクトリに溜まらないこと。"""
  parent = param_dir.parent
  for _ in range(3):
    lcp.write(lcp.KEY_OFFSET, 0.1)
  assert [f for f in os.listdir(parent) if f.startswith('.tmp_value_')] == []
  assert [f for f in os.listdir(param_dir) if f.startswith('.tmp')] == []


def test_write_replaces_atomically(param_dir, monkeypatch):
  """差し替えは rename 一発で、truncate → write の隙間を作らないこと。

  ⚠ 普通に open('w') すると、controlsd がその瞬間に読むと**空 = 既定値 (OFF)** を掴む。
  走行中に補正が一瞬切れることになるので、rename であることを固定しておく。
  """
  (param_dir / lcp.KEY_OFFSET).write_bytes(b"0.30")

  seen = {}
  real_replace = os.replace

  def spy(src, dst):
    # rename の直前でも、読み手には**古い中身がそのまま**見えていること
    seen['before'] = (param_dir / lcp.KEY_OFFSET).read_bytes()
    return real_replace(src, dst)

  monkeypatch.setattr(os, 'replace', spy)
  assert lcp.write(lcp.KEY_OFFSET, -0.1)
  assert seen['before'] == b"0.30"
  assert (param_dir / lcp.KEY_OFFSET).read_bytes() == b"-0.1"


def test_write_failure_is_reported(tmp_path, monkeypatch):
  """書けなかったら False。⚠ UI はこれを見て表示を元に戻す (画面 ON / 制御 OFF を作らない)。

  ⚠ 保存先は初回書き込みで `os.makedirs(exist_ok=True)` するようになったので、「親が無い」
  だけでは失敗しない。`d` の位置に**ファイル**を置いて makedirs を落としている。
  """
  blocked = tmp_path / "blocked"
  blocked.write_bytes(b"not a directory")
  monkeypatch.setattr(lcp, 'PARAM_DIR', str(blocked))
  assert lcp.write(lcp.KEY_ENABLED, True) is False


def test_first_write_creates_the_save_location(tmp_path, monkeypatch):
  """⚠ `/data/params` と違って openpilot の誰も作ってくれないので、最初の書き込みで作ること。

  作られないと UI が毎回「書けなかった」と言って、設定が一切保存できない。
  """
  fresh = tmp_path / "params_fork" / "d"
  monkeypatch.setattr(lcp, 'PARAM_DIR', str(fresh))
  assert not fresh.exists()
  assert lcp.write(lcp.KEY_OFFSET, 0.1)
  assert (fresh / lcp.KEY_OFFSET).read_bytes() == b"0.1"
  # ⚠ tmp と .lock は `d` の親に作るので、親も一緒に出来ていること
  assert fresh.parent.is_dir()


def test_controller_reads_what_the_ui_wrote(param_dir):
  """UI が書いた 4 つを制御側がそのまま拾えること (この 2 つが繋がっていることが本題)。"""
  lcp.write(lcp.KEY_ENABLED, True)
  lcp.write(lcp.KEY_OFFSET, -0.2)
  lcp.write(lcp.KEY_AUTHORITY, 0.5)
  lcp.write(lcp.KEY_PAUSE_ON_SIGNAL, False)

  c = lc.LaneCenteringController()
  assert c.enabled is True
  assert c.offset == pytest.approx(-0.2)
  assert c.e2e_authority == pytest.approx(0.5)
  assert c.pause_on_signal is False
