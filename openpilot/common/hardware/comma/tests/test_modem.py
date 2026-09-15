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
