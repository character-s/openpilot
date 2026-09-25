"""stall_watchdog (GS450h) の挙動テスト。標準ライブラリだけで回る (fork の CI でも走らせる)。

実スレッドで「beat が止まる → 戻る」を起こし、レポートに ①止まった関数の Python スタック ②止まっていた秒数
が載ること、平常時は何も書かないことを見る。/proc の読み取りは Linux のときだけ確かめる。
"""
import ctypes
import os
import sys
import threading
import time

import pytest

from openpilot.system.hardware import stall_watchdog
from openpilot.system.hardware.stall_watchdog import StallWatchdog, sample_task


class Recorder:
  def __init__(self):
    self.events: list[tuple[str, dict]] = []
    self.reports: list[str] = []
    self.dmesg = 0

  def event(self, name, **kw):
    self.events.append((name, kw))

  def write(self, text):
    self.reports.append(text)
    return f'/fake/hwstall-{len(self.reports)}.log'

  def save_dmesg(self):
    self.dmesg += 1

  def names(self):
    return [n for n, _ in self.events]


def _make(rec: Recorder, **kw) -> StallWatchdog:
  opts = {'threshold': 0.4, 'rearm_interval': 0.05, 'poll': 0.02, 'gap_log_threshold': 0.2}
  opts.update(kw)
  return StallWatchdog('hardwared', log_event=rec.event, write_report=rec.write, save_dmesg=rec.save_dmesg, **opts)


def _beat_for(wd: StallWatchdog, seconds: float):
  end = time.monotonic() + seconds
  while time.monotonic() < end:
    wd.beat()
    time.sleep(0.01)


def blocking_call_marker(seconds: float):
  # レポートの faulthandler 出力にこの関数名が出ることを確かめる
  time.sleep(seconds)


def blocking_with_gil_marker(seconds: float):
  # PyDLL 経由の呼び出しは GIL を離さない = 「GIL を握ったまま C の中で止まる」を再現する
  if sys.platform == 'win32':
    ctypes.PyDLL('kernel32').Sleep(int(seconds * 1000))
  else:
    ctypes.PyDLL(None).usleep(int(seconds * 1e6))


def _run_loop(wd: StallWatchdog, stalls: list[float], block=blocking_call_marker):
  def loop():
    _beat_for(wd, 0.2)
    for s in stalls:
      block(s)
      _beat_for(wd, 0.3)
  t = threading.Thread(target=loop)
  t.start()
  t.join()
  time.sleep(0.2)  # 監視スレッドが「戻った」を拾って書き出すのを待つ


def test_stall_is_reported_with_python_stack_and_duration():
  rec = Recorder()
  wd = _make(rec)
  wd.start()
  try:
    _run_loop(wd, [1.0])
  finally:
    wd.stop()

  assert len(rec.reports) == 1, rec.names()
  report = rec.reports[0]
  assert 'blocking_call_marker' in report, 'faulthandler の Python スタックに止まった関数が出ていない'
  assert 'gap:' in report
  stall = [kw for n, kw in rec.events if n == 'hardwared stall']
  assert len(stall) == 1 and 0.9 <= stall[0]['gap'] <= 1.5, stall
  assert stall[0]['report'] == '/fake/hwstall-1.log'
  assert 'hardwared stall detected' in rec.names(), '止まっている最中のイベント (電源断に備えた先出し) が無い'
  assert rec.dmesg == 1


def test_stall_while_holding_gil_is_still_reported():
  """監視スレッドも止まる (GIL を握られる) ケース。faulthandler の C スレッドが書いたスタックを事後に回収できること。"""
  rec = Recorder()
  wd = _make(rec)
  wd.start()
  try:
    _run_loop(wd, [1.0], block=blocking_with_gil_marker)
  finally:
    wd.stop()

  assert len(rec.reports) == 1, rec.names()
  assert 'blocking_with_gil_marker' in rec.reports[0], 'GIL 保持中の停止で Python スタックを取れていない'
  assert 'GIL を握ったまま止まっていた' in rec.reports[0]
  stall = [kw for n, kw in rec.events if n == 'hardwared stall']
  assert len(stall) == 1 and stall[0]['gil_held'] is True and 0.9 <= stall[0]['gap'] <= 1.5, stall


def test_report_is_written_even_if_traceback_buffer_cannot_be_opened(monkeypatch):
  monkeypatch.setattr(StallWatchdog, '_open_tb_file', staticmethod(lambda: None))
  rec = Recorder()
  wd = _make(rec)
  wd.start()
  try:
    _run_loop(wd, [0.7])
  finally:
    wd.stop()
  assert len(rec.reports) == 1 and '出力先を開けなかった' in rec.reports[0]


def test_healthy_loop_writes_nothing():
  rec = Recorder()
  wd = _make(rec)
  wd.start()
  try:
    t = threading.Thread(target=_beat_for, args=(wd, 1.0))
    t.start()
    t.join()
    time.sleep(0.1)
  finally:
    wd.stop()
  assert rec.reports == [] and rec.dmesg == 0
  assert not [n for n in rec.names() if 'stall' in n], rec.names()


def test_short_gap_is_logged_without_report():
  rec = Recorder()
  wd = _make(rec)
  wd.start()
  try:
    _run_loop(wd, [0.25])  # gap_log_threshold (0.2) 以上・threshold (0.4) 未満
  finally:
    wd.stop()
  assert rec.reports == []
  gaps = [kw['gap'] for n, kw in rec.events if n == 'hardwared loop gap']
  assert len(gaps) == 1 and 0.2 <= gaps[0] < 0.4, rec.events


def test_report_count_is_capped():
  rec = Recorder()
  wd = _make(rec, max_reports=1)
  wd.start()
  try:
    _run_loop(wd, [0.7, 0.7])
  finally:
    wd.stop()
  assert len(rec.reports) == 1, 'max_reports を超えてファイルを書いた (止まり続けたら crash 置き場が埋まる)'
  assert [kw.get('note') for n, kw in rec.events if n == 'hardwared stall'] == [None, 'max_reports']


@pytest.mark.skipif(not sys.platform.startswith('linux'), reason='/proc が要る')
def test_sample_task_sees_a_thread_blocked_in_read():
  r, w = os.pipe()
  tid = []

  def reader():
    tid.append(threading.get_native_id())
    os.read(r, 1)

  t = threading.Thread(target=reader)
  t.start()
  try:
    time.sleep(0.2)
    s = sample_task(tid[0])
    assert s['state'] == 'S', s
    assert s['syscall'].split()[0].isdigit(), s  # syscall 中 (running / -1 ではない)
    if stall_watchdog.AARCH64:
      assert s['syscall_name'] == 'read' and s['fd_target'].startswith('pipe:'), s
  finally:
    os.write(w, b'x')
    t.join()
    os.close(r)
    os.close(w)
