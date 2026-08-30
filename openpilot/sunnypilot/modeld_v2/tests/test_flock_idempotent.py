import fcntl
import os

from openpilot.common.test import OpenpilotTestCase

from tinygrad.helpers import temp
from tinygrad.runtime.support.system import System


class TestFlockIdempotent(OpenpilotTestCase):
  """`System.flock_acquire` が同じプロセスからの再取得で自分に弾かれないこと。

  ⚠⚠ 08-30 の実車ハングの核心。flock は fd ではなく **open file description** 単位で排他する
  ので、同じプロセスが `os.open` をやり直して掛け直すと **自分自身が持っているロック**とぶつかる。
  実測では 1 プロセスが 6 回 `flock_acquire` に来て成功は最初の 1 回だけ、残り 5 回は自分に
  弾かれて fd を 5 個リークしていた。おかげで「デバイス初期化がなぜ失敗したのか」が毎回
  「ロックが取れない」に化け、リトライは原理的に成功しない状態だった。
  """

  @staticmethod
  def _release_all():
    """System は singleton。掴んだロックを解放して cached_property も捨てる。"""
    for fd in System.__dict__.get("_held_locks", {}).values():
      try:
        os.close(fd)
      except OSError:
        pass
    System.__dict__.pop("_held_locks", None)

  @staticmethod
  def _fd_count() -> int:
    return len(os.listdir("/proc/self/fd"))

  def setUp(self):
    self._release_all()
    self.addCleanup(self._release_all)

  def test_same_name_twice_returns_same_fd(self):
    name = "test_flock_idem_a.lock"
    self.assertEqual(System.flock_acquire(name), System.flock_acquire(name))

  def test_no_fd_leak_on_repeated_acquire(self):
    name = "test_flock_idem_b.lock"
    System.flock_acquire(name)
    before = self._fd_count()
    for _ in range(5):
      System.flock_acquire(name)
    self.assertEqual(self._fd_count(), before)

  def test_other_holder_raises_without_leaking(self):
    """他の open file description が握っていれば失敗する = プロセス間の排他は保つ。

    ⚠ 失敗した fd を閉じないと、以後の試行が全部その fd に弾かれる (これが自家中毒の入口)。
    """
    name = "test_flock_idem_c.lock"
    other = os.open(temp(name), os.O_RDWR | os.O_CREAT, 0o666)
    self.addCleanup(os.close, other)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    before = self._fd_count()
    with self.assertRaisesRegex(RuntimeError, "Failed to acquire lock file"):
      System.flock_acquire(name)
    self.assertEqual(self._fd_count(), before, "失敗した fd が閉じられていない")

  def test_recovers_after_other_holder_releases(self):
    """保持者が手放したら取れる = 失敗を握り込まない。"""
    name = "test_flock_idem_d.lock"
    other = os.open(temp(name), os.O_RDWR | os.O_CREAT, 0o666)
    fcntl.flock(other, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with self.assertRaises(RuntimeError):
      System.flock_acquire(name)
    os.close(other)
    self.assertGreater(System.flock_acquire(name), 0)

  def test_flock_is_per_open_file_description(self):
    """道具の前提そのものを固定する。同一プロセスでも **別の os.open なら排他される**。

    ⚠ この性質を知らないと「自分が握っているのに自分が取れない」が理解できない。
    """
    path = temp("test_flock_idem_e.lock")
    a, b = os.open(path, os.O_RDWR | os.O_CREAT, 0o666), os.open(path, os.O_RDWR)
    self.addCleanup(os.close, a)
    self.addCleanup(os.close, b)
    fcntl.flock(a, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with self.assertRaises(BlockingIOError):
      fcntl.flock(b, fcntl.LOCK_EX | fcntl.LOCK_NB)
