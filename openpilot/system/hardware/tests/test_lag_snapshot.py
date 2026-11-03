"""lag_snapshot (GS450h) の判定テスト。標準ライブラリだけで回る。

11-02 route 122 のように selfdrivedLagging が続いたブートでは 1 回だけ取り、平常のブートでは比べる相手 (baseline) を
1 回だけ取ることを見る。書き出しの中身は c4 の実機で確かめる (debugfs / sudo が要る)。
"""
import os

from openpilot.system.hardware import lag_snapshot as ls
from openpilot.system.hardware.lag_snapshot import LagSnapshot


class Recorder:
  def __init__(self):
    self.events: list[tuple[str, dict]] = []
    self.captures: list[tuple[str, dict]] = []

  def event(self, name, **kw):
    self.events.append((name, kw))

  def capture(self, kind, out_dir, summary):
    self.captures.append((kind, summary))
    return f'{out_dir}/{kind}'


def _make(rec: Recorder, tmp_path) -> LagSnapshot:
  return LagSnapshot(log_event=rec.event, out_dir=str(tmp_path), capture_fn=rec.capture, background=False)


def _drive(snap: LagSnapshot, lag_pattern: list[bool], start: float = 1000., onroad_since: float | None = 1000., dt: float = 0.5):
  t = start
  for lagging in lag_pattern:
    snap.update(t, onroad_since, lagging)
    t += dt
  return t


def test_short_single_lag_is_ignored(tmp_path):
  """起動直後などの一瞬の遅れ 1 回では取らない。"""
  rec = Recorder()
  snap = _make(rec, tmp_path)
  _drive(snap, [False] * 4 + [True] * 4 + [False] * 4)
  assert [k for k, _ in rec.captures] == []


def test_repeated_lag_takes_one_snapshot(tmp_path):
  rec = Recorder()
  snap = _make(rec, tmp_path)
  pattern = ([True, True, False, False] * ls.LAG_ONSETS_BEFORE_SNAPSHOT) * 3
  _drive(snap, pattern)
  assert [k for k, _ in rec.captures] == ['lag'], '同じブートでは 1 回だけ'
  assert rec.captures[0][1]['lag_onsets'] == ls.LAG_ONSETS_BEFORE_SNAPSHOT
  assert rec.events[0][0] == 'lag snapshot'


def test_continuous_lag_takes_snapshot_by_seconds(tmp_path):
  """立ち上がりが 1 回でも出っぱなしなら秒数で取る (122 は出っぱなしに近かった)。"""
  rec = Recorder()
  snap = _make(rec, tmp_path)
  _drive(snap, [True] * int(ls.LAG_SECONDS_BEFORE_SNAPSHOT / 0.5 + 2))
  assert [k for k, _ in rec.captures] == ['lag']
  assert rec.captures[0][1]['lag_onsets'] == 1


def test_baseline_once_after_onroad(tmp_path):
  rec = Recorder()
  snap = _make(rec, tmp_path)
  n = int(ls.BASELINE_AFTER_ONROAD / 0.5)
  _drive(snap, [False] * (n - 1))
  assert rec.captures == [], 'onroad 直後には取らない'
  _drive(snap, [False] * (n * 3), start=1000. + (n - 1) * 0.5)
  assert [k for k, _ in rec.captures] == ['baseline'], '平常時の値は 1 ブート 1 回'


def test_offroad_does_not_count(tmp_path):
  """offroad の間の selfdriveState は古い値なので数えない。"""
  rec = Recorder()
  snap = _make(rec, tmp_path)
  _drive(snap, [True, False] * 20, onroad_since=None)
  assert rec.captures == [] and snap.lag_onsets == 0


def test_capture_failure_does_not_raise(tmp_path):
  """hardwared を落とさない: 書き出しが例外を投げてもイベントに残すだけ。"""
  rec = Recorder()

  def boom(kind, out_dir, summary):
    raise OSError('disk full')

  snap = LagSnapshot(log_event=rec.event, out_dir=str(tmp_path), capture_fn=boom, background=False)
  _drive(snap, [True] * int(ls.LAG_SECONDS_BEFORE_SNAPSHOT / 0.5 + 2))
  assert rec.events and rec.events[0][0] == 'lag snapshot failed'


def test_latest_route_by_number_not_mtime(tmp_path):
  """起動直後は時計が 07-28 に化けて新しい route の方が古い時刻になる (11-02 実機で 123 を拾った)。"""
  for name, mtime in (('00000123--8dc76c1a5b--27', 2000), ('00000124--6f1246d6c1--0', 1000),
                      ('00000124--6f1246d6c1--1', 1001), ('boot', 3000), ('crash', 3000)):
    (tmp_path / name).mkdir()
    os.utime(tmp_path / name, (mtime, mtime))
  assert ls._latest_route(str(tmp_path)) == '00000124--6f1246d6c1--1'


def test_prune_keeps_newest_by_mtime(tmp_path):
  """名前ではなく更新時刻で古いものを消す (起動直後は日時が 07-28 に化ける)。"""
  for i in range(ls.KEEP + 3):
    d = tmp_path / f'name{(ls.KEEP + 3 - i):02d}'  # 名前の順と作った順を逆にする
    d.mkdir()
    os.utime(d, (1000 + i, 1000 + i))
  ls.prune(str(tmp_path))
  left = sorted(p.name for p in tmp_path.iterdir())
  assert len(left) == ls.KEEP
  assert 'name01' in left and f'name{ls.KEEP + 3:02d}' not in left
