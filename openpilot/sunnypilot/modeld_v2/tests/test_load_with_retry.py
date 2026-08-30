from unittest import mock

from openpilot.common.test import OpenpilotTestCase
import openpilot.sunnypilot.modeld_v2.modeld as modeld


LOCK_ERR = "Failed to acquire lock file am_usb:4-2.lock"


class TestLoadWithRetry(OpenpilotTestCase):
  """eGPU のロード失敗が「ロック競合」に化けて本当の理由を隠さないこと。

  ⚠ 08-30 実車: 10 連敗のログが全部 `lock holder = 自分自身` で、なぜ GPU を掴めなかったのかが
  1 件も残っていなかった。tinygrad は候補インターフェースを順に試し、1 つ目でロックを取ったまま
  失敗すると 2 つ目以降が自分の残したロックとぶつかるため。
  """

  def setUp(self):
    for name in ("_egpu_lock_holder",):
      p = mock.patch.object(modeld, name, return_value="nobody")
      p.start()
      self.addCleanup(p.stop)
    for name in ("warning", "error"):
      p = mock.patch.object(modeld.cloudlog, name)
      self.addCleanup(p.stop)
      setattr(self, f"log_{name}", p.start())

  def test_returns_model_on_success(self):
    model, err = modeld._load_with_retry(lambda: "MODEL", attempts=3, wait=0)
    self.assertEqual(model, "MODEL")
    self.assertIsNone(err)

  def test_returns_first_error_not_last(self):
    """リトライ後の「ロックが取れない」ではなく、1 回目に起きたことを返す。"""
    errs = [RuntimeError(f"real reason: gpu not responding ({LOCK_ERR})"), RuntimeError(LOCK_ERR)]

    def make_model():
      raise errs.pop(0) if len(errs) > 1 else errs[0]

    model, err = modeld._load_with_retry(make_model, attempts=3, wait=0)
    self.assertIsNone(model)
    self.assertIn("real reason", repr(err))

  def test_does_not_retry_when_not_lock_contention(self):
    """ロック競合でないなら 1 回で諦める (待っても状況は変わらない)。"""
    calls = []

    def make_model():
      calls.append(1)
      raise RuntimeError("gpu is on fire")

    model, err = modeld._load_with_retry(make_model, attempts=5, wait=0)
    self.assertIsNone(model)
    self.assertEqual(len(calls), 1)
    self.assertIn("on fire", repr(err))

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
