"""chestnut_power_limit のテスト。

⚠ 実機の SMU には触れない。`apply_power_limit` が**関数の中で** import する `tinygrad.device` を
`sys.modules` ごと差し替えて、送ったメッセージの順序と読み戻しだけを見る。

⚠ このモジュールが「絞る値」の唯一の出所なので、**既定値と上下限が設計点から外れていないこと**も
ここで機械的に押さえる (170W という stock の異常値に引きずられて上限を上げてしまわないため)。
"""
import ast
import sys
import types
from pathlib import Path

import pytest

from openpilot.sunnypilot.modeld_v2 import chestnut_power_limit as cpl


@pytest.fixture
def param_dir(tmp_path, monkeypatch):
  d = tmp_path / "d"
  d.mkdir()
  monkeypatch.setattr(cpl, 'PARAM_DIR', str(d))
  return d


class _FakeSmuMod:
  PPSMC_MSG_SetPptLimit = 0x11
  PPSMC_MSG_GetPptLimit = 0x12


class _FakeSmu:
  """SMU の最小の身代わり。Set で内部値を更新し、Get(read_back_arg=True) でそれを返す。"""

  def __init__(self, limit=170, fail=False):
    self.smu_mod = _FakeSmuMod
    self.limit = limit
    self.calls: list = []
    self.fail = fail

  def _send_msg(self, msg, arg, read_back_arg=False, timeout=None):
    if self.fail:
      raise RuntimeError("SMU msg 0x11 timeout")
    self.calls.append((msg, arg))
    if msg == _FakeSmuMod.PPSMC_MSG_SetPptLimit:
      self.limit = arg
    return self.limit if read_back_arg else 0


@pytest.fixture
def fake_smu(monkeypatch):
  """`tinygrad` と `tinygrad.device` を両方差し替える。

  ⚠ 片方だけだと `from tinygrad.device import Device` が親パッケージを実 import しに行く。
  """
  def _install(limit=170, fail=False):
    smu = _FakeSmu(limit, fail)
    device_mod = types.ModuleType("tinygrad.device")
    device_mod.Device = {"AMD": types.SimpleNamespace(iface=types.SimpleNamespace(dev_impl=types.SimpleNamespace(smu=smu)))}
    tinygrad_mod = types.ModuleType("tinygrad")
    tinygrad_mod.device = device_mod
    monkeypatch.setitem(sys.modules, "tinygrad", tinygrad_mod)
    monkeypatch.setitem(sys.modules, "tinygrad.device", device_mod)
    return smu
  return _install


# ── 値の読み取り ──────────────────────────────────────────────

def test_default_caps_when_no_file(param_dir):
  """⚠ ファイルが無い = **絞る**。openpilot の Params 既定と違い「未設定なら stock」にはしない。"""
  assert cpl.get_power_limit() == cpl.DEFAULT_LIMIT_W == 80


def test_file_value_wins(param_dir):
  (param_dir / cpl.KEY_POWER_LIMIT).write_bytes(b"60")
  assert cpl.get_power_limit() == 60


def test_zero_means_stock(param_dir):
  """0 = 絞らない。⚠ これが性能低下時の退避先なので消してはいけない。"""
  (param_dir / cpl.KEY_POWER_LIMIT).write_bytes(b"0")
  assert cpl.get_power_limit() == 0


@pytest.mark.parametrize("written,expected", [(b"10", 40), (b"39", 40), (b"150", 100), (b"101", 100)])
def test_clamped_to_design_range(param_dir, written, expected):
  (param_dir / cpl.KEY_POWER_LIMIT).write_bytes(written)
  assert cpl.get_power_limit() == expected


@pytest.mark.parametrize("written", [b"", b"   ", b"abc", b"80W"])
def test_garbage_falls_back_to_default(param_dir, written):
  (param_dir / cpl.KEY_POWER_LIMIT).write_bytes(written)
  assert cpl.get_power_limit() == cpl.DEFAULT_LIMIT_W


def test_whitespace_and_float_text_are_accepted(param_dir):
  (param_dir / cpl.KEY_POWER_LIMIT).write_bytes(b" 60.0\n")
  assert cpl.get_power_limit() == 60


# ── 定義の健全性 ──────────────────────────────────────────────

def test_range_stays_within_the_design_point():
  """chestnut の推論設計点は 100W。stock が 170W を返すからといって上限を上げない。"""
  assert cpl.POWER_LIMIT_MAX_W == 100
  assert cpl.POWER_LIMIT_MIN_W == 40
  assert cpl.POWER_LIMIT_MIN_W <= cpl.DEFAULT_LIMIT_W <= cpl.POWER_LIMIT_MAX_W


def test_default_actually_reduces_the_stock_limit():
  """既定が stock (実測 170W / 上流の 9060 は 160W) より小さいこと = 絞る側であること。"""
  assert cpl.DEFAULT_LIMIT_W < 160


# ── SMU への適用 ──────────────────────────────────────────────

def test_zero_touches_nothing(fake_smu):
  smu = fake_smu()
  assert cpl.apply_power_limit(0) is None
  assert smu.calls == []


def test_sets_then_reads_back(fake_smu):
  smu = fake_smu(limit=170)
  assert cpl.apply_power_limit(80) == 80
  # Set が先、Get が後。順序が逆だと「効いた値」ではなく古い値を読む。
  assert [msg for msg, _ in smu.calls] == [_FakeSmuMod.PPSMC_MSG_SetPptLimit, _FakeSmuMod.PPSMC_MSG_GetPptLimit]
  assert smu.calls[0][1] == 80


def test_smu_failure_does_not_raise(fake_smu):
  """⚠ 絞れなくても load は続行する (big を粘る方針)。例外が漏れると modeld が死ぬ。"""
  fake_smu(fail=True)
  assert cpl.apply_power_limit(80) is None


# ── #1964 の再現防止 ──────────────────────────────────────────

def test_tinygrad_is_not_imported_at_module_scope():
  """⚠⚠ モジュールスコープで tinygrad に触らないこと。

  issue #1964 = `compile_modeld.py` が `Device.DEFAULT` を既定引数に置いたせいで **import 時に**
  AMD デバイスを開き、flock を握ったまま失敗して本番の load を自分で塞いだ。同じ形をここで作らない。
  """
  assert "tinygrad" not in _module_scope_import_roots()


def test_module_scope_imports_stay_in_the_stdlib():
  """⚠ このテスト自体が PC で回る状態を守る見張り。

  `openpilot.common.swaglog` は zmq を引くので、モジュールスコープで import すると
  **PC では import すらできなくなる** (`lane_centering_params.py` が同じ罠を docstring で警告している)。
  ログは `_log()` の中で遅延 import すること。
  """
  roots = _module_scope_import_roots()
  assert roots <= {"os"}, f"module-scope imports must stay stdlib-only, got {sorted(roots)}"


def _module_scope_import_roots() -> set[str]:
  """モジュール直下 (= import 時に必ず走る位置) の import のトップレベル名。

  ⚠ body 直下だけ見る = 関数の中の import は対象外 (それが正しい置き方)。
  """
  tree = ast.parse(Path(cpl.__file__).read_text(encoding="utf-8"))
  roots: set[str] = set()
  for node in tree.body:
    if isinstance(node, ast.Import):
      roots.update(a.name.split('.')[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
      roots.add((node.module or "").split('.')[0])
  return roots
