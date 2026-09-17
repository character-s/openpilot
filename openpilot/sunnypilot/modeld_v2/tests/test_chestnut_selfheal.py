"""chestnut_selfheal のテスト。

⚠⚠ **実車で自分から再起動をかけるコード**なので、暴走しない証明をここに集める。
モジュール docstring の「4 重の歯止め」を 1 つずつ機械で押さえる:
  ①最初の失敗から 300s 粘る (回数だけでは撃たない) ②この起動では 1 回だけ
  ③累計 2 回で打ち止め (**再起動を跨いで効くこと**) ④`ChestnutAutoReboot=0` で止まる
⚠ ③ は「`/dev/shm` に置いたら再起動のたびにカウンタが消えて無限ループ」という一番危ない穴の番人。
  `REBOOT_COUNT_PATH` が消えない場所であることを前提にしたテストを必ず残すこと。
"""
import json
import sys
import types

import pytest

from openpilot.sunnypilot.modeld_v2 import chestnut_selfheal as sh


@pytest.fixture
def paths(tmp_path, monkeypatch):
  """ブート状態・累計・スイッチの置き場を全部 tmp に逃がす。"""
  param_dir = tmp_path / "d"
  param_dir.mkdir()
  monkeypatch.setattr(sh, 'PARAM_DIR', str(param_dir))
  monkeypatch.setattr(sh, 'BOOT_STATE_PATH', str(tmp_path / "chestnut_selfheal.json"))
  monkeypatch.setattr(sh, 'REBOOT_COUNT_PATH', str(tmp_path / "chestnut_selfheal_reboots"))
  return types.SimpleNamespace(
    param_dir=param_dir,
    boot=tmp_path / "chestnut_selfheal.json",
    count=tmp_path / "chestnut_selfheal_reboots",
  )


@pytest.fixture
def fake_params(monkeypatch):
  """`openpilot.common.params` を差し替えて、立てた params を覗けるようにする。

  ⚠ PC には capnp が無く本物は import できない。⚠ 親パッケージ側も置かないと
  `from openpilot.common.params import Params` が実 import に行く。
  """
  put: dict = {}

  class _Params:
    def put_bool(self, k, v):
      put[k] = v

  mod = types.ModuleType("openpilot.common.params")
  mod.Params = _Params
  monkeypatch.setitem(sys.modules, "openpilot.common.params", mod)
  return put


def _fail_until_reboot(t0=1000.0, step=60.0, limit=60):
  """再起動すべきと判定されるまで失敗を入れ、(時刻, 回数) を返す。"""
  t = t0
  for i in range(1, limit + 1):
    if sh.note_load_failure(now=t):
      return t, i
    t += step
  raise AssertionError("再起動の判定に至らなかった")


class TestHold1_PersistLongEnough:
  def test_single_failure_does_not_reboot(self, paths):
    assert sh.note_load_failure(now=1000.0) is False

  def test_many_failures_in_a_short_window_do_not_reboot(self, paths):
    """⚠ 回数では撃たない (08-30 の「粘れば戻る」の教訓)。"""
    t = 1000.0
    for _ in range(20):
      assert sh.note_load_failure(now=t) is False
      t += 1.0

  def test_long_window_alone_does_not_reboot(self, paths):
    assert sh.note_load_failure(now=1000.0) is False
    assert sh.note_load_failure(now=1000.0 + sh.REBOOT_AFTER_SEC + 1) is False
    assert sh.MIN_FAILS == 3

  def test_reboots_after_persistent_failure(self, paths):
    t, fails = _fail_until_reboot()
    assert fails >= sh.MIN_FAILS
    assert t - 1000.0 >= sh.REBOOT_AFTER_SEC


class TestSelfLockShortcut:
  """★ 09-17 の症状 (#1964 の自家中毒) では、粘る時間の中身は **今までと同じソフトリセット** で
  効かないと分かっている ⇒ 短い閾値に切り替わること。"""

  # 09-17 の実ログそのまま (swaglog の `eGPU model load failed (lock holder: ...)`)
  HOLDER_SELF = "my_fds=['16'] flock_held_by=['59551(me:openpilot.selfd:HELD)'] fuser=59551 comma F.... openpilot.selfd"
  HOLDER_OTHER = "my_fds=[] flock_held_by=['1234(other:someproc:HELD)'] fuser=1234"
  HOLDER_WAIT = "my_fds=['16'] flock_held_by=['59551(me:openpilot.selfd:WAIT)'] fuser=59551"
  HOLDER_NONE = "my_fds=[] flock_held_by=nobody fuser=nobody"

  def test_detects_self_held_lock(self):
    assert sh.is_self_locked(self.HOLDER_SELF) is True

  @pytest.mark.parametrize("holder", [HOLDER_OTHER, HOLDER_WAIT, HOLDER_NONE, "", "unknown"])
  def test_other_cases_are_not_self_locked(self, holder):
    """⚠ `:WAIT` は「ロック待ち」で保持者ではない。他人が持っているのも別の話。"""
    assert sh.is_self_locked(holder) is False

  def test_fuser_alone_does_not_count(self):
    """⚠ `fuser` は「開いているだけ」= 保持とは限らない (tinygrad は open してから flock する)。"""
    assert sh.is_self_locked("flock_held_by=[] fuser=59551 (me:openpilot.selfd:HELD)") is False

  def test_self_lock_reboots_sooner(self, paths):
    """09-17 の時系列 (39s → 118s) を当てると、3 周目で発火する = 約 2 分。"""
    assert sh.note_load_failure(now=1000.0, self_locked=True) is False
    assert sh.note_load_failure(now=1039.0, self_locked=True) is False   # 39s: 回数は足りるが時間が足りない
    assert sh.note_load_failure(now=1118.0, self_locked=True) is True    # 118s > 60s
    assert sh.REBOOT_AFTER_SEC_SELF_LOCK == 60.0
    assert sh.MIN_FAILS_SELF_LOCK == 2

  def test_same_timeline_without_self_lock_keeps_waiting(self, paths):
    """⚠ 原因が読めないときは従来どおり 300s 粘る (GPU ハングなら粘れば戻るため)。"""
    for t in (1000.0, 1039.0, 1118.0, 1281.0):
      assert sh.note_load_failure(now=t) is False
    assert sh.note_load_failure(now=1000.0 + sh.REBOOT_AFTER_SEC) is True

  def test_self_lock_is_sticky(self, paths):
    """⚠ 周回ごとに holder が読めたり読めなかったりする。一度観測したら短い方を使い続ける。"""
    assert sh.note_load_failure(now=1000.0, self_locked=True) is False
    assert sh.note_load_failure(now=1070.0, self_locked=False) is True   # 読めなくても 60s 側で見る

  def test_self_lock_still_honours_the_switch(self, paths):
    (paths.param_dir / sh.KEY_AUTO_REBOOT).write_bytes(b"0")
    for t in (1000.0, 1100.0, 1200.0):
      assert sh.note_load_failure(now=t, self_locked=True) is False


class TestHold2_OncePerBoot:
  def test_no_second_request_in_the_same_boot(self, paths, fake_params):
    _fail_until_reboot()
    assert sh.request_reboot() is True
    for t in (5000.0, 6000.0, 7000.0):
      assert sh.note_load_failure(now=t) is False, "この起動では 1 回しか撃たない"

  def test_flag_is_written_before_the_reboot_is_requested(self, paths, fake_params):
    """⚠ 順序が逆だと、再起動が間に合ったとき「打った」が残らず毎起動で撃つ。"""
    _fail_until_reboot()
    sh.request_reboot()
    assert json.loads(paths.boot.read_text())['rebooted'] is True
    assert paths.count.read_text().strip() == "1"


class TestHold3_MaxReboots:
  def test_count_survives_a_boot(self, paths, fake_params):
    """★ ここが無限ループの番人 — 再起動で消える場所に累計を置いてはいけない。"""
    _fail_until_reboot()
    sh.request_reboot()
    paths.boot.unlink()                      # 再起動 = /dev/shm が消える
    assert paths.count.exists(), "累計は再起動を跨いで残らなければならない"
    assert sh.reboot_count() == 1

  def test_stops_at_max_reboots(self, paths, fake_params):
    for expected in (1, 2):
      _fail_until_reboot()
      assert sh.request_reboot() is True
      assert sh.reboot_count() == expected
      paths.boot.unlink()                    # 次の起動

    # 3 回目は撃たない (= 再起動では直らない相手だと判断する)
    t = 1000.0
    for _ in range(20):
      assert sh.note_load_failure(now=t) is False
      t += 60.0
    assert sh.reboot_count() == sh.MAX_REBOOTS == 2

  def test_success_clears_the_count(self, paths, fake_params):
    _fail_until_reboot()
    sh.request_reboot()
    paths.boot.unlink()
    sh.note_load_success()                   # 再起動後に big が載った
    assert sh.reboot_count() == 0
    assert not paths.count.exists()

    # 累計が戻ったので、次の事故ではまた再起動を使える
    _fail_until_reboot()
    assert sh.request_reboot() is True

  def test_no_reboot_when_the_count_cannot_be_persisted(self, paths, fake_params, monkeypatch):
    """⚠ 累計が書けないなら撃たない。歯止め 3 が効かないまま撃つと無限ループになる。"""
    monkeypatch.setattr(sh, 'REBOOT_COUNT_PATH', str(paths.count / "no" / "such" / "dir"))
    _fail_until_reboot()
    assert sh.request_reboot() is False
    assert "DoReboot" not in fake_params


class TestHold4_Switch:
  def test_enabled_by_default(self, paths):
    assert sh.auto_reboot_enabled() is True

  @pytest.mark.parametrize("raw", [b"0", b"false", b"False", b""])
  def test_disabled_by_param(self, paths, raw):
    (paths.param_dir / sh.KEY_AUTO_REBOOT).write_bytes(raw)
    assert sh.auto_reboot_enabled() is False

  def test_disabled_param_blocks_the_reboot(self, paths):
    (paths.param_dir / sh.KEY_AUTO_REBOOT).write_bytes(b"0")
    t = 1000.0
    for _ in range(20):
      assert sh.note_load_failure(now=t) is False
      t += 60.0

  @pytest.mark.parametrize("raw", [b"1", b"true", b"yes"])
  def test_other_values_keep_it_enabled(self, paths, raw):
    (paths.param_dir / sh.KEY_AUTO_REBOOT).write_bytes(raw)
    assert sh.auto_reboot_enabled() is True


class TestRebootRequest:
  def test_sets_do_reboot(self, paths, fake_params):
    _fail_until_reboot()
    assert sh.request_reboot() is True
    assert fake_params == {"DoReboot": True}, "manager が拾う params を立てるだけ"

  def test_params_failure_is_contained(self, paths, monkeypatch):
    """⚠ params が立てられなくても例外にしない (呼び出し元はこの後 raise して死ぬ)。"""
    broken = types.ModuleType("openpilot.common.params")

    class _Broken:
      def put_bool(self, k, v):
        raise RuntimeError("no params")
    broken.Params = _Broken
    monkeypatch.setitem(sys.modules, "openpilot.common.params", broken)

    _fail_until_reboot()
    assert sh.request_reboot() is False


class TestRobustness:
  def test_corrupt_boot_state_is_ignored(self, paths):
    paths.boot.write_text("{ not json")
    assert sh.note_load_failure(now=1000.0) is False

  def test_garbage_fail_count_does_not_raise(self, paths):
    paths.boot.write_text(json.dumps({'first_fail_mono': 0.0, 'fails': 'lots'}))
    assert sh.note_load_failure(now=1.0) is False
    assert json.loads(paths.boot.read_text())['fails'] == 1

  def test_future_origin_is_reset(self, paths):
    """⚠ 古い起点を信じると起動直後の 1 回目でいきなり再起動してしまう。"""
    paths.boot.write_text(json.dumps({'first_fail_mono': 99999.0, 'fails': 9}))
    assert sh.note_load_failure(now=10.0) is False
    assert json.loads(paths.boot.read_text())['first_fail_mono'] == 10.0

  def test_corrupt_count_reads_as_zero(self, paths):
    paths.count.write_text("garbage")
    assert sh.reboot_count() == 0

  def test_success_without_any_state_is_quiet(self, paths):
    sh.note_load_success()   # 何も無くても例外にしない
