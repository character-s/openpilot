from unittest import TestCase, mock

import openpilot.common.hardware.comma.modem as modem_mod
from openpilot.common.hardware.comma.modem import Modem, SIM_READ_FAILS_BEFORE_RESET


class TestSimRadioReset(TestCase):
  """GS450h: 起動時に SIM を掴み損ねたら、リトライだけで粘らずにラジオを入れ直す。

  09-15 に実車で 2 回踏んだ: AT+CPIN? が +CME ERROR: 13 (SIM failure) を返し続け、
  _read_identity() が永久に失敗して state が INITIALIZING のまま 37 分動かなかった。
  AT+CFUN=0/1 を手で叩くと +CPIN: READY に戻り、そのまま CONNECTED まで進むのを確認している。
  """

  def _modem(self) -> Modem:
    # __init__ は PPPSession を作って実機のポートを触りに行くので通さない
    m = Modem.__new__(Modem)
    m._sim_read_fails = 0
    m._at = mock.Mock()
    return m

  @staticmethod
  def _at_commands(m: Modem) -> list[str]:
    return [c.args[0] for c in m._at.call_args_list]

  def test_does_not_reset_before_threshold(self):
    """起動直後の数回はモデム側の初期化待ちなので、すぐには打たない。"""
    m = self._modem()
    with mock.patch.object(modem_mod.time, "sleep"):
      for _ in range(SIM_READ_FAILS_BEFORE_RESET - 1):
        m._reset_radio_if_sim_unreadable()
    self.assertEqual(self._at_commands(m), [])

  def test_resets_at_threshold(self):
    m = self._modem()
    with mock.patch.object(modem_mod.time, "sleep"):
      for _ in range(SIM_READ_FAILS_BEFORE_RESET):
        m._reset_radio_if_sim_unreadable()
    self.assertEqual(self._at_commands(m), ["AT+CFUN=0", "AT+CFUN=1"])

  def test_resets_again_after_another_run(self):
    """一度入れ直しても駄目なら、また同じ間隔で入れ直す (諦めない)。"""
    m = self._modem()
    with mock.patch.object(modem_mod.time, "sleep"):
      for _ in range(SIM_READ_FAILS_BEFORE_RESET * 2):
        m._reset_radio_if_sim_unreadable()
    self.assertEqual(self._at_commands(m), ["AT+CFUN=0", "AT+CFUN=1"] * 2)

  def test_waits_between_off_and_on(self):
    """CFUN=0 の直後に 1 を打つと SIM が立ち上がらないので、必ず待つ。"""
    m = self._modem()
    with mock.patch.object(modem_mod.time, "sleep") as sleep:
      for _ in range(SIM_READ_FAILS_BEFORE_RESET):
        m._reset_radio_if_sim_unreadable()
    self.assertEqual([c.args[0] for c in sleep.call_args_list],
                     [modem_mod.RADIO_OFF_WAIT, modem_mod.RADIO_ON_WAIT])

  def test_counter_is_cleared_on_success(self):
    """identity が読めたら数え直す (次の失敗が即リセットにならないように)。"""
    m = self._modem()
    with mock.patch.object(modem_mod.time, "sleep"):
      for _ in range(SIM_READ_FAILS_BEFORE_RESET - 1):
        m._reset_radio_if_sim_unreadable()
      m._sim_read_fails = 0  # _do_initializing が成功したときの挙動
      m._reset_radio_if_sim_unreadable()
    self.assertEqual(self._at_commands(m), [])


class TestSearchRadioReset(TestCase):
  """GS450h: SIM は読めているのに登録が取れず SEARCHING で止まったら、ラジオを入れ直す。

  11-02 に 2 ブート続けて踏んだ: identity も APN も通って SEARCHING に入ったあと +CREG / +CGREG が
  登録済みにならず、1 回目は走行中ずっと、2 回目は 13 分そのままだった (普段は 1-46 秒で登録される)。
  """

  def _modem(self, creg: str, cgreg: str) -> Modem:
    m = Modem.__new__(Modem)
    m.S = modem_mod.INITIAL_STATE.copy()
    m._roaming_allowed = True
    m._sim_change = False
    m._search_since = None
    m._last_reg = None
    replies = {"AT+CREG?": f"+CREG: 2,{creg}", "AT+CGREG?": f"+CGREG: 2,{cgreg}"}
    m._at = mock.Mock(side_effect=lambda cmd: [replies[cmd]] if cmd in replies else [])
    m._publish_state = mock.Mock()
    m._is_roaming_allowed = mock.Mock(return_value=True)
    return m

  @staticmethod
  def _at_commands(m: Modem) -> list[str]:
    return [c.args[0] for c in m._at.call_args_list if c.args[0].startswith("AT+CFUN")]

  def _search(self, m: Modem, at: float):
    with mock.patch.object(modem_mod.time, "monotonic", return_value=at), \
         mock.patch.object(modem_mod.time, "sleep"), \
         mock.patch.object(modem_mod.os.path, "exists", return_value=True):
      return m._do_searching()

  def test_does_not_reset_while_searching_normally(self):
    m = self._modem("2", "2")  # 2 = searching
    m._search_since = 1000.0
    self.assertEqual(self._search(m, 1000.0 + modem_mod.SEARCH_TIMEOUT_BEFORE_RESET - 1), modem_mod.State.SEARCHING)
    self.assertEqual(self._at_commands(m), [])

  def test_resets_after_timeout(self):
    m = self._modem("2", "2")
    m._search_since = 1000.0
    after = 1000.0 + modem_mod.SEARCH_TIMEOUT_BEFORE_RESET
    self.assertEqual(self._search(m, after), modem_mod.State.SEARCHING)
    self.assertEqual(self._at_commands(m), ["AT+CFUN=0", "AT+CFUN=1"])
    # 入れ直したら数え直す (毎ループ叩き続けない)
    self._search(m, after + 1)
    self.assertEqual(self._at_commands(m), ["AT+CFUN=0", "AT+CFUN=1"])

  def test_resets_when_creg_does_not_answer(self):
    m = self._modem("2", "2")
    m._at.side_effect = lambda cmd: []
    m._search_since = 1000.0
    self._search(m, 1000.0 + modem_mod.SEARCH_TIMEOUT_BEFORE_RESET)
    self.assertEqual(self._at_commands(m), ["AT+CFUN=0", "AT+CFUN=1"])

  def test_registered_goes_to_connecting_without_reset(self):
    m = self._modem("1", "1")  # 1 = home
    m._search_since = 1000.0
    self.assertEqual(self._search(m, 1000.0 + modem_mod.SEARCH_TIMEOUT_BEFORE_RESET * 2), modem_mod.State.CONNECTING)
    self.assertEqual(self._at_commands(m), [])

  def test_roaming_refused_by_setting_is_not_reset(self):
    """ローミング禁止で待っているのは設定どおりなので、入れ直しても意味がない。"""
    m = self._modem("5", "5")  # 5 = roaming
    m._is_roaming_allowed.return_value = False
    m._roaming_allowed = False
    m._search_since = 1000.0
    self.assertEqual(self._search(m, 1000.0 + modem_mod.SEARCH_TIMEOUT_BEFORE_RESET * 2), modem_mod.State.SEARCHING)
    self.assertEqual(self._at_commands(m), [])
