from unittest import mock

from openpilot.common.test import OpenpilotTestCase
import openpilot.system.manager.process as process_mod
from openpilot.system.manager.process import ManagerProcess


class _Dead:
  """終了済みプロセスの代役 (exitcode が None なら「まだ生きている」)。"""

  def __init__(self, exitcode: int | None = 1):
    self.exitcode = exitcode


class _Proc(ManagerProcess):
  def __init__(self):
    self.name = "modeld_tinygrad"
    self.restart_on_crash = True
    self.crash_count = 0
    self.last_crash_t = 0.0
    self.last_start_t = 0.0
    self.proc = None
    self.shutting_down = False

  def start(self) -> None:
    self.proc = _Dead()
    self.last_start_t = process_mod.time.monotonic()


class TestRestartOnCrash(OpenpilotTestCase):
  """chestnut の GPU ハングから big model のまま復帰させる再起動ロジック。

  ⚠⚠ 08-30 実車の実測が前提: 30-60 秒間隔の再ロードは **10 回連続で `no pcie`** に終わり、
  **4 分 28 秒空けた 1 回**で復帰した。失敗と成功の違いは **間隔だけ**。
  「n 回で諦める」設計はこの復帰を永久に取り逃すので、回数では打ち切らない。
  """

  def setUp(self):
    self.now = 1000.0
    patcher = mock.patch.object(process_mod.time, "monotonic", side_effect=lambda: self.now)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.p = _Proc()

  def _crash(self) -> bool:
    """落ちた状態にして reap を試みる。掃除されて再起動できる状態になったら True。"""
    self.p.proc = _Dead()
    with mock.patch.object(process_mod.cloudlog, "error"):
      self.p.reap_if_crashed()
    return self.p.proc is None

  def test_first_crash_is_reaped_immediately(self):
    self.assertTrue(self._crash())
    self.assertEqual(self.p.crash_count, 1)

  def test_waits_backoff_before_next_restart(self):
    self.assertTrue(self._crash())
    self.now += 5.0                    # 2 回目の待ち (20s) には足りない
    self.assertFalse(self._crash())
    self.assertEqual(self.p.crash_count, 1)
    self.now += 20.0
    self.assertTrue(self._crash())
    self.assertEqual(self.p.crash_count, 2)

  def test_backoff_doubles(self):
    for count, expected in ((0, 10.0), (1, 20.0), (2, 40.0), (3, 80.0), (4, 160.0)):
      self.p.crash_count = count
      self.assertEqual(self.p.restart_backoff(), expected)

  def test_backoff_is_capped(self):
    self.p.crash_count = 20
    self.assertEqual(self.p.restart_backoff(), self.p.RESTART_BACKOFF_MAX)

  def test_never_gives_up(self):
    """回数で打ち切らない。08-30 は 10 連敗のあとの試行で復帰した。"""
    reaped = 0
    for _ in range(20):
      self.now += self.p.RESTART_BACKOFF_MAX + 1
      if self._crash():
        reaped += 1
    self.assertEqual(reaped, 20)

  def test_forgets_count_after_living_long_enough(self):
    """一度まともに動けたなら、次のクラッシュは 1 回目として扱う (待ちが伸びたままにしない)。"""
    self.p.crash_count = 4
    self.p.last_start_t = self.now
    self.p.last_crash_t = self.now
    self.now += self.p.CRASH_FORGET + 1
    self.assertTrue(self._crash())
    self.assertEqual(self.p.crash_count, 1)

  def test_does_not_forget_while_never_recovering(self):
    """一度も復帰できていないのにカウントが戻ると、待ちがいつまでも伸びない。"""
    self.assertTrue(self._crash())
    self.p.last_start_t = 0.0           # 起動できていない = 生きた時間が無い
    self.now += self.p.CRASH_FORGET + 1
    self.assertTrue(self._crash())
    self.assertEqual(self.p.crash_count, 2)

  def test_ignores_when_restart_on_crash_is_off(self):
    self.p.restart_on_crash = False
    self.assertFalse(self._crash())

  def test_ignores_while_still_running(self):
    self.p.proc = _Dead(exitcode=None)
    with mock.patch.object(process_mod.cloudlog, "error"):
      self.p.reap_if_crashed()
    self.assertIsNotNone(self.p.proc)

  def test_ignores_while_shutting_down(self):
    self.p.shutting_down = True
    self.assertFalse(self._crash())
