import io
import sys
import tempfile
from collections import deque
from pathlib import Path
from unittest import mock

from openpilot.common.test import OpenpilotTestCase
import openpilot.selfdrive.modeld.helpers as helpers


class TestStdioCapture(OpenpilotTestCase):
  """chestnut の GPU ハング時に、tinygrad が print する割り込み情報を拾えること (機序は helpers._StdioTee の docstring)。"""

  def setUp(self):
    self._saved = (sys.stdout, sys.stderr, helpers._STDIO_RING, helpers.CRASH_DIR)
    helpers._STDIO_RING = None

  def tearDown(self):
    sys.stdout, sys.stderr, helpers._STDIO_RING, helpers.CRASH_DIR = self._saved

  def test_tee_passes_through(self):
    orig = io.StringIO()
    ring: deque = deque(maxlen=10)
    tee = helpers._StdioTee(orig, ring)
    print("hello", file=tee)
    self.assertEqual(orig.getvalue(), "hello\n")
    self.assertEqual(list(ring), ["hello"])

  def test_partial_line_kept_until_newline(self):
    ring: deque = deque(maxlen=10)
    tee = helpers._StdioTee(io.StringIO(), ring)
    tee.write("abc")
    self.assertEqual(list(ring), [])
    self.assertEqual(tee.pending(), "abc")
    tee.write("def\nghi")
    self.assertEqual(list(ring), ["abcdef"])
    self.assertEqual(tee.pending(), "ghi")

  def test_partial_line_flushed_when_too_long(self):
    """改行の来ない出力でメモリを食い続けない。"""
    ring: deque = deque(maxlen=10)
    tee = helpers._StdioTee(io.StringIO(), ring)
    tee.write("x" * 5000)
    self.assertEqual(len(ring), 1)
    self.assertEqual(tee.pending(), "")

  def test_broken_stream_still_records(self):
    """元のストリームが死んでいても記録は続ける (落ちる寸前に呼ばれるため)。"""
    class Dead:
      def write(self, s):
        raise OSError("gone")

    ring: deque = deque(maxlen=10)
    tee = helpers._StdioTee(Dead(), ring)
    tee.write("still recorded\n")
    self.assertEqual(list(ring), ["still recorded"])

  def test_capture_stdio_is_idempotent(self):
    """二重に包むと出力が二重に流れるので、2 回目は何もしない。"""
    helpers.capture_stdio()
    first = sys.stdout
    helpers.capture_stdio()
    self.assertIs(sys.stdout, first)

  def test_capture_stdio_never_raises(self):
    """落ちる寸前の保険が、modeld の起動を妨げてはいけない。"""
    with (mock.patch.object(helpers, "_StdioTee", side_effect=RuntimeError("boom")),
          mock.patch.object(helpers.cloudlog, "exception")):
      helpers.capture_stdio()
    self.assertIsNone(helpers._STDIO_RING)

  def test_snapshot_without_capture_is_noop(self):
    self.assertIsNone(helpers.save_stdio_snapshot("-crash"))

  def test_snapshot_writes_file_and_reports_gpu_faults(self):
    with tempfile.TemporaryDirectory() as td:
      helpers.CRASH_DIR = Path(td)
      helpers.capture_stdio()
      print("am 0000:01:00.0: sq_intr: error MEMVIOL")
      print("nothing interesting")
      with mock.patch.object(helpers.cloudlog, "error") as err, mock.patch.object(helpers.cloudlog, "warning"):
        path = helpers.save_stdio_snapshot("-crash")

      self.assertIsNotNone(path)
      body = path.read_text()
      self.assertIn("MEMVIOL", body)
      self.assertIn("nothing interesting", body)

    err.assert_called_once()
    self.assertIn("MEMVIOL", err.call_args[0][0])

  def test_snapshot_keeps_unterminated_last_line(self):
    """改行前に落ちた最後の 1 行こそ知りたいので、pending も書き出す。"""
    with tempfile.TemporaryDirectory() as td:
      helpers.CRASH_DIR = Path(td)
      helpers.capture_stdio()
      sys.stdout.write("UTCL2_FAULT no newline yet")
      with mock.patch.object(helpers.cloudlog, "error"), mock.patch.object(helpers.cloudlog, "warning"):
        path = helpers.save_stdio_snapshot("-crash")

      self.assertIn("UTCL2_FAULT no newline yet", path.read_text())
