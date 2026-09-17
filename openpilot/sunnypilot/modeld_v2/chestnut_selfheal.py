"""big が載らないまま粘り続けたら、c4 が**自分で再起動して big を取り直す** (GS450h 追加)。

⚠⚠ **なぜ要るのか (09-17 の実車)**:
走行中に big の load が **#1964 の flock 自家中毒** (import 時の `Device.DEFAULT` probe が自分で
ロックを握る) で失敗し始め、「起動 → 10 秒で失敗 → `reset_chestnut()` → crash → manager が再起動」を
**30-150 秒おきに 45 分繰り返した**。`restart_on_crash` は倍々 backoff で**回数では諦めない**ので、
**この構造からはソフトだけでは永久に抜けられない**。実際の復旧は **c4 の再起動**だった
(ロックはプロセスと一緒に消える) が、それを人が SSH で叩くまで走れなかった。
⇒ user 09-17:「毎回 opus を呼び出すのも面倒だね」 ⇒ **c4 が自分で同じ手を打つ**。

⚠⚠ **small (RHM) への降格は採らない** — user 09-17:**「small はいらない」**。
big と small の実力差が大きく、RHM で走るくらいなら openpilot を使わないという判断
([[user_driving_model]] の 09-17)。⇒ **復帰の方向は「big のまま取り直す」= 再起動しかない**。

⚠⚠⚠ **効くのは「ロックを握ったまま」のときだけ**。GPU が本当に Device hang している場合は
chestnut が XT60 から別給電されていて **c4 を落としても GPU の電源は切れない** ので再起動は効かない
(09-15 実測: 再起動 2 分後に再発)。⇒ **だから上限を置いて 2 回で諦める** (下記 `MAX_REBOOTS`)。
「効かない相手に再起動を繰り返す」のが一番たちが悪い。

**再起動が暴走しないための 4 重の歯止め**:
1. **最初の失敗から `REBOOT_AFTER_SEC` 粘る** — cold boot の一過性の失敗では発火しない。
   ⚠ 回数だけで切らないのは 08-30 の教訓 (「ハングは粘れば戻る — 失敗と成功の違いは間隔だけ」で
   `ea6809f316`「再起動の諦めを効かせる」を撤回した経緯)。回数は誤爆防止の下限に留める。
2. **この起動では 1 回だけ** (`/dev/shm` のフラグ = 再起動で消える)。
3. **累計 `MAX_REBOOTS` 回で打ち止め** (`/data` に残す = **再起動を跨いで効く**)。
   ⚠⚠ `/dev/shm` だけだと**再起動のたびにカウンタが消えて無限ループになる**。ここが一番危ない所。
4. **`ChestnutAutoReboot` に 0 を書けば止まる** (c4 上でファイルを置くだけ・再配信不要)。

⚠ 再起動そのものは **`DoReboot` params を立てるだけ**。manager がメインループで拾って
全プロセスを畳んでから `HARDWARE.reboot()` する (`system/manager/manager.py:185`) = openpilot の作法。
⚠ `DoReboot` は `CLEAR_ON_MANAGER_START` なので、**起動のたびに自動で消える**。

⚠ **安全**: ここが呼ばれる = big の load に失敗している = `modelV2` が出ていない ⇒
`bigModelLoading` / `bigModelFailed` で **engage できない**。つまり**再起動で失う制御は無い**
(手動運転はそのまま続く)。⚠ modeld は `only_onroad` なので **offroad でこの経路には入らない**。

⚠ **モジュールスコープの依存は標準ライブラリだけに保つ** (`chestnut_power_limit.py` と同じ約束)。
PC には zmq が無く `openpilot.common.swaglog` も `Params` も import できないため、関数の中で引く。
"""
import json
import os
import re
import time

# ⚠ テストが monkeypatch する差し替え点なので、参照側は `from ... import X` ではなく
#    **モジュール属性で** 引くこと (`chestnut_power_limit.py` と同じ約束)。
PARAM_DIR = '/data/params_fork/d'
KEY_AUTO_REBOOT = "ChestnutAutoReboot"

# この起動での失敗の起点と回数。⚠ tmpfs = 電源断と再起動で必ず消える (歯止め 2)。
BOOT_STATE_PATH = '/dev/shm/chestnut_selfheal.json'

# 自動再起動の累計。⚠⚠ **再起動を跨いで残る場所に置くこと** (歯止め 3)。
# `/dev/shm` に置くと再起動のたびに 0 に戻り、直らない相手に永久に再起動をかけ続ける。
REBOOT_COUNT_PATH = '/data/chestnut_selfheal_reboots'

# 最初の load 失敗からこれだけ経っても載らなければ再起動する。
# ⚠ **この待ち時間の中身は「今までと同じソフトリセット」** = modeld の 5 回リトライ →
# `reset_chestnut()` (USB reset) → crash → manager の倍々 backoff で再起動、の繰り返し。
# 実測の間隔は 39s → 79s → 163s (09-17) なので 300s は **3-4 周**に当たる。
# 粘る価値があるのは **GPU の Device hang** のとき — 08-30 に「粘れば戻る。失敗と成功の違いは
# 間隔だけ」と分かって `ea6809f316`「再起動の諦めを効かせる」を撤回した経緯がある。
REBOOT_AFTER_SEC = 300.0
MIN_FAILS = 3

# ⚠⚠ ただし **自分で flock を握ったまま弾かれている (#1964 の自家中毒)** と分かっているときは話が別。
# modeld の再起動も USB reset も **modeld 自身が毎サイクルやっていることと同じ** ⇒ 何周しても戻らない
# (09-17 は 45 分抜けられなかった)。この症状に効くのは**ロックをプロセスごと消す再起動だけ**なので、
# 粘る時間を短くする。⇒ 09-17 の時系列なら **初回失敗の約 2 分後**に自力で戻れていた。
# ⚠ 誤爆しても歯止め 3 (累計 2 回) で止まる。
REBOOT_AFTER_SEC_SELF_LOCK = 60.0
MIN_FAILS_SELF_LOCK = 2

# `_egpu_lock_holder()` が出す `flock_held_by=['<pid>(me:<comm>:HELD)']` を拾う。
# ⚠ `:WAIT` は「ロック待ち」で保持者ではない (潰して読むと待っている自分を保持者と誤読する)。
# ⚠ `fuser=` の側は「開いているだけ」で保持とは限らないので、**`flock_held_by` の中だけ**を見る。
_HELD_BY_RE = re.compile(r"flock_held_by=\[([^\]]*)\]")
_SELF_HELD_RE = re.compile(r"\(me:[^)]*:HELD\)")

# これ以上やっても直らない = GPU 側の Device hang なので人を待つ (歯止め 3)。
MAX_REBOOTS = 2


def is_self_locked(holder: str) -> bool:
  """`_egpu_lock_holder()` の文字列が「**自分が flock を保持している**」と言っているか。

  ⚠ 読めなければ False = 長く粘る側 (安全側) に倒す。判定に失敗したせいで
  「効かないソフトリセットを 5 分続ける」方に倒れるのは許容するが、逆は許容しない。
  """
  m = _HELD_BY_RE.search(holder or "")
  return bool(m and _SELF_HELD_RE.search(m.group(1)))


def _read_json(path: str) -> dict:
  """⚠ 例外を投げない。状態が読めないことでモデルのロードを止めては本末転倒。"""
  try:
    with open(path) as f:
      state = json.load(f)
  except Exception:
    return {}
  return state if isinstance(state, dict) else {}


def _write_json(path: str, state: dict) -> None:
  """⚠ 書けなくても黙って続行する。⚠ 半分書けたものを読ませないよう tmp 経由で置き換える。"""
  try:
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
      json.dump(state, f)
    os.replace(tmp, path)
  except OSError:
    pass


def _log(**kwargs) -> None:
  """cloudlog へ 1 行残す。⚠ 遅延 import + 失敗を握る (モジュール docstring 参照)。

  **再起動したことは必ず残す**。再起動で dmesg は消えるので、後から
  「なぜ落ちたのか」を追えるのはここと `/data/community/crashes/` だけになる。
  """
  try:
    from openpilot.common.swaglog import cloudlog
    cloudlog.event("chestnut selfheal", **kwargs)
  except Exception:
    pass


def auto_reboot_enabled() -> bool:
  """⚠ 既定は **有効**。`ChestnutAutoReboot` に `0` を書いたときだけ止まる (歯止め 4)。

  ⚠ 置き場が `/data/params_fork/d` なのは `chestnut_power_limit.py` と同じ理由 —
  `common/params.cc` の `clearAll` がホワイトリスト外のキーを manager 起動のたびに消すため。
  ⇒ **c4 上でファイルを置くだけで止められる** (再配信も `.so` の焼き直しも要らない)。
  """
  try:
    with open(os.path.join(PARAM_DIR, KEY_AUTO_REBOOT), 'rb') as f:
      raw = f.read().strip()
  except OSError:
    return True
  return raw not in (b'0', b'false', b'False', b'')


def reboot_count() -> int:
  """これまでに自動再起動した回数。⚠ big が載ったら 0 に戻る (`note_load_success`)。"""
  try:
    with open(REBOOT_COUNT_PATH) as f:
      return max(0, int(f.read().strip() or 0))
  except (OSError, ValueError):
    return 0


def note_load_failure(now: float | None = None, self_locked: bool = False) -> bool:
  """big の load 失敗を 1 回記録する。**この失敗で「再起動すべき」が確定したら True**。

  `self_locked` = 自分で flock を握ったまま弾かれていたか (`is_self_locked`)。真なら閾値を短くする。

  ⚠ 時刻は `time.monotonic()` = CLOCK_MONOTONIC (ブート起点・プロセス間で共通)。壁時計は使わない —
  c4 の RTC は既定値で固定され GPS の unix 時刻も不正 ([[reference_c4_rlog_no_wallclock]])。
  ⇒ ブートで 0 に戻るが、起点ごと tmpfs から消えるので整合する。
  """
  now = time.monotonic() if now is None else now
  state = _read_json(BOOT_STATE_PATH)

  first = state.get('first_fail_mono')
  # ⚠ 起点が壊れている / 未来になっているなら取り直す。古い起点を信じると起動直後の 1 回目で発火する。
  if not isinstance(first, int | float) or first > now:
    first = now

  try:
    fails = int(state.get('fails', 0)) + 1
  except (TypeError, ValueError):
    # ⚠ 記録が壊れていても判定は続ける。ここで例外を投げると「粘り続けて走れない」状態に戻る。
    fails = 1

  elapsed = now - first
  already = bool(state.get('rebooted'))
  # ⚠ 一度でも自家中毒を観測したら以降も短い閾値で見る。周回ごとに holder が読めたり読めなかったり
  #    するので、**観測できた事実の方を残す** (読めない = False に引きずられて粘らせない)。
  sticky_self_lock = bool(state.get('self_locked')) or bool(self_locked)
  _write_json(BOOT_STATE_PATH, {'first_fail_mono': first, 'fails': fails,
                                'rebooted': already, 'self_locked': sticky_self_lock})

  need_sec = REBOOT_AFTER_SEC_SELF_LOCK if sticky_self_lock else REBOOT_AFTER_SEC
  need_fails = MIN_FAILS_SELF_LOCK if sticky_self_lock else MIN_FAILS

  if already:
    return False                       # 歯止め 2: この起動ではもう打った
  if not (fails >= need_fails and elapsed >= need_sec):
    return False                       # 歯止め 1: まだ粘る
  if not auto_reboot_enabled():
    _log(action="skipped", why="disabled", fails=fails, elapsed_s=round(elapsed, 1),
         self_locked=sticky_self_lock)
    return False                       # 歯止め 4
  if reboot_count() >= MAX_REBOOTS:
    # 歯止め 3: 再起動で直らない相手 (= GPU の Device hang) に繰り返さない。
    _log(action="skipped", why="max reboots reached", reboots=reboot_count(),
         fails=fails, elapsed_s=round(elapsed, 1), self_locked=sticky_self_lock,
         note="likely a real GPU hang; power-cycle the XT60")
    return False
  return True


def note_load_success() -> None:
  """big が載ったときに呼ぶ。連続失敗の記録と**再起動の累計**を捨てる。

  ⚠ 累計をここで 0 に戻すのが歯止め 3 の要 — 「1 回の再起動で直った」なら次の事故でもまた
  再起動を使えるが、「再起動しても直らない」ときは累計が減らないので 2 回で止まる。
  """
  try:
    os.remove(BOOT_STATE_PATH)
  except OSError:
    pass
  if reboot_count():
    _log(action="recovered", reboots=reboot_count())
  try:
    os.remove(REBOOT_COUNT_PATH)
  except OSError:
    pass


def request_reboot(now: float | None = None) -> bool:
  """`DoReboot` を立てて manager に再起動させる。**立てられたら True**。

  ⚠ ここで `HARDWARE.reboot()` を直接呼ばない。manager がメインループで `DoReboot` を拾い、
  **全プロセスを畳んでから**再起動する (`system/manager/manager.py:185`) のが openpilot の作法。
  ⚠ 先に累計とフラグを**書いてから**立てる。逆順だと、間に合わずに再起動が走ったとき
  「この起動で打った」が残らず、**起動のたびに再起動する無限ループ**になる。
  """
  state = _read_json(BOOT_STATE_PATH)
  state['rebooted'] = True
  _write_json(BOOT_STATE_PATH, state)

  count = reboot_count() + 1
  try:
    with open(REBOOT_COUNT_PATH, 'w') as f:
      f.write(str(count))
      f.flush()
      os.fsync(f.fileno())   # ⚠ 直後に電源が飛ぶ前提の書き込みなので fsync する
  except OSError:
    # ⚠ 累計が書けないなら **再起動しない**。歯止め 3 が効かないまま再起動すると無限ループになる。
    _log(action="aborted", why="cannot persist reboot count", path=REBOOT_COUNT_PATH)
    return False

  _log(action="reboot", reboots=count, max_reboots=MAX_REBOOTS,
       note="big model did not come back; rebooting to drop the stale flock")
  try:
    from openpilot.common.params import Params
    Params().put_bool("DoReboot", True)
    return True
  except Exception as e:
    _log(action="failed", exc=repr(e))
    return False
