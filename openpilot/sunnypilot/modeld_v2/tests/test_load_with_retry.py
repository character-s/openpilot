from unittest import mock

from openpilot.common.test import OpenpilotTestCase
import openpilot.sunnypilot.modeld_v2.modeld as modeld


LOCK_ERR = "Failed to acquire lock file am_usb:4-2.lock"


def _lock_error() -> ExceptionGroup:
  # shape of the error tinygrad raises when the am_usb flock is held by another process
  return ExceptionGroup('No interface for AMD:0 is available',
                        [FileNotFoundError(2, 'No such file or directory'), RuntimeError('no pcie'), RuntimeError(LOCK_ERR)])


class TestLoadWithRetry(OpenpilotTestCase):
  """eGPU のロード失敗が「ロック競合」に化けて本当の理由を隠さないこと (機序は modeld._load_with_retry の docstring)。"""

  def setUp(self):
    p = mock.patch.object(modeld, "_egpu_lock_holder", return_value="nobody")
    p.start()
    self.addCleanup(p.stop)
    for name in ("warning", "error"):
      p = mock.patch.object(modeld.cloudlog, name)
      self.addCleanup(p.stop)
      setattr(self, f"log_{name}", p.start())

  def test_returns_model_on_success(self):
    make_model = mock.Mock(return_value="MODEL")
    model, err = modeld._load_with_retry(make_model, attempts=3, wait=0)
    self.assertEqual((model, err), ("MODEL", None))
    self.assertEqual(make_model.call_count, 1)

  def test_returns_first_error_not_last(self):
    """リトライ後の「ロックが取れない」ではなく、1 回目に起きたことを返す。"""
    make_model = mock.Mock(side_effect=[RuntimeError(f"real reason: gpu not responding ({LOCK_ERR})"),
                                        RuntimeError(LOCK_ERR), RuntimeError(LOCK_ERR)])
    model, err = modeld._load_with_retry(make_model, attempts=3, wait=0)
    self.assertIsNone(model)
    self.assertIn("real reason", repr(err))

  def test_does_not_retry_when_not_lock_contention(self):
    """ロック競合でないなら 1 回で諦める (待っても状況は変わらない)。"""
    make_model = mock.Mock(side_effect=RuntimeError("gpu is on fire"))
    model, err = modeld._load_with_retry(make_model, attempts=5, wait=0)
    self.assertIsNone(model)
    self.assertEqual(make_model.call_count, 1)
    self.assertIn("on fire", repr(err))

  def test_lock_contention_detected(self):
    self.assertTrue(modeld._is_lock_contention(_lock_error()))
    self.assertTrue(modeld._is_lock_contention(RuntimeError('Failed to acquire lock file am_usb:2-1.lock')))
    self.assertFalse(modeld._is_lock_contention(RuntimeError('no pcie')))
    self.assertFalse(modeld._is_lock_contention(RuntimeError("args mismatch in JIT: captured=(..., 'QCOM') expected=(..., 'AMD')")))

  def test_gives_up_after_all_attempts(self):
    make_model = mock.Mock(side_effect=_lock_error())
    model, err = modeld._load_with_retry(make_model, wait=0)
    self.assertIsNone(model)
    self.assertEqual(make_model.call_count, modeld.EGPU_LOAD_ATTEMPTS)
    self.assertTrue(modeld._is_lock_contention(err))

  def test_success_after_contention(self):
    make_model = mock.Mock(side_effect=[_lock_error(), _lock_error(), "MODEL"])
    model, err = modeld._load_with_retry(make_model, wait=0)
    self.assertEqual((model, err), ("MODEL", None))
    self.assertEqual(make_model.call_count, 3)

  def test_logs_full_traceback_of_first_failure(self):
    """ExceptionGroup は repr だと 1 行に潰れるので、1 回目だけ traceback を残す。"""
    def make_model():
      raise RuntimeError(LOCK_ERR)

    modeld._load_with_retry(make_model, attempts=2, wait=0)
    logged = " ".join(str(c) for c in self.log_error.call_args_list)
    self.assertIn("attempt 1 failed", logged)
    self.assertIn("Traceback", logged)

  def test_logs_lock_holder_before_loading(self):
    """起動時点で誰かが握っているのか、自分で取って自分で失敗したのかを分けるため。"""
    modeld._load_with_retry(lambda: "MODEL", attempts=1, wait=0)
    logged = " ".join(str(c) for c in self.log_warning.call_args_list)
    self.assertIn("lock holder before", logged)
