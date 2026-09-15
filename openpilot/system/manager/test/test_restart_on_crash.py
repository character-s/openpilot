from types import SimpleNamespace
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
import openpilot.system.manager.process as process_mod
from openpilot.system.manager.process import ManagerProcess, PythonProcess


class _Proc(ManagerProcess):
  """restart_on_crash だけ ON。他は ManagerProcess のクラス属性既定値 (proc=None, count=0 ...) のまま。"""
  name = "modeld_tinygrad"
  restart_on_crash = True

  def start(self) -> None:  # abstractmethod の穴埋め。テストは proc を直接差し替える
    pass


class TestRestartOnCrash(OpenpilotTestCase):
  """chestnut の GPU ハングから big model のまま復帰させる再起動ロジック (回数で打ち切らず、待ちを倍にする)。"""

  def setUp(self):
    self.now = 1000.0
    patcher = mock.patch.object(process_mod.time, "monotonic", side_effect=lambda: self.now)
    patcher.start()
    self.addCleanup(patcher.stop)
    self.p = _Proc()

  def _crash(self) -> bool:
    """落ちた状態にして reap を試みる。掃除されて再起動できる状態になったら True。"""
    self.p.proc = SimpleNamespace(exitcode=1)
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
    self.p.proc = SimpleNamespace(exitcode=None)  # exitcode None = まだ生きている
    with mock.patch.object(process_mod.cloudlog, "error"):
      self.p.reap_if_crashed()
    self.assertIsNotNone(self.p.proc)

  def test_ignores_while_shutting_down(self):
    self.p.shutting_down = True
    self.assertFalse(self._crash())


class TestPythonProcessRestartOnCrash(OpenpilotTestCase):
  """GS450h: PythonProcess でも restart_on_crash が効く (locationd の stuck からの復帰に使う)。

  09-15 に locationd を kill したら二度と上がらず、c4 の再起動でしか戻せなかった。
  locationd は modeld (poll='cameraOdometry') の死に道連れになり inputsOK=False のまま
  復帰しないことがあるので、自分で落ちて manager に作り直させる経路を用意する。
  """

  @staticmethod
  def _proc(**kwargs) -> PythonProcess:
    return PythonProcess("locationd", "openpilot.selfdrive.locationd.locationd", lambda *a: True, **kwargs)

  def _start(self, p: PythonProcess):
    with mock.patch.object(process_mod, "Process") as mock_process, \
         mock.patch.object(process_mod.cloudlog, "error"), \
         mock.patch.object(process_mod.cloudlog, "info"):
      p.start()
    return mock_process.called

  def test_flag_defaults_to_off(self):
    self.assertFalse(self._proc().restart_on_crash)

  def test_flag_can_be_enabled(self):
    self.assertTrue(self._proc(restart_on_crash=True).restart_on_crash)

  def test_start_reaps_crashed_proc(self):
    """crash 済みの proc が残っていても start() が掃除して起動し直す (upstream はここで詰まる)。"""
    p = self._proc(restart_on_crash=True)
    p.proc = SimpleNamespace(exitcode=1)
    self.assertTrue(self._start(p))

  def test_start_does_not_reap_without_the_flag(self):
    """既定 (False) のプロセスの挙動は変えない = 落ちたままにする。"""
    p = self._proc()
    p.proc = SimpleNamespace(exitcode=1)
    self.assertFalse(self._start(p))

  def test_start_is_noop_while_running(self):
    """生きているプロセスを二重起動しない。"""
    p = self._proc(restart_on_crash=True)
    p.proc = SimpleNamespace(exitcode=None)
    self.assertFalse(self._start(p))

  def test_locationd_is_configured_to_restart(self):
    """process_config 側で実際に有効になっていること (ここが外れると kill したきり戻らない)。"""
    from openpilot.system.manager.process_config import managed_processes
    self.assertTrue(managed_processes["locationd"].restart_on_crash)
