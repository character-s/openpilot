"""driving personality の 4 段トグル (mici)。GS450h: moreAggressive @3 を足したための専用版。

⚠ `BigMultiParamToggle` をそのまま 4 段で使うと見た目が崩れる:
  - pill は `y += 35` で縦に積むので 4 個目が y=105..171 になり、ボタン下端の角と value 行
    ("more aggressive") に食い込む。
  - 並び = Params の保存値 (enum 番号) なので、上から aggressive / standard / relaxed / more aggressive に
    なり、一番詰める段が一番下に来る。
  そこで「表示の並び」と「保存値」を分け、pill は上流 3 段と同じ縦幅 (0..136) に 4 個詰める。

⚠ 新規ファイルにしているのは `widgets/button.py` が upstream 側のファイルで、追従のたびに衝突しうるため
  (lane_centering.py と同じ方針)。
"""
from openpilot.common.params import Params
from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle

# 表示の並び (上から = 車間の短い順)。保存値は enum 番号 (aggressive=0, standard=1, relaxed=2, moreAggressive=3)。
PERSONALITY_LABELS = ("more aggressive", "aggressive", "standard", "relaxed")
PERSONALITY_VALUES = (3, 0, 1, 2)

# 上流 3 段の pill が占める縦幅 (2 * 35 + 66 = 136px) に 4 段を収める間隔。
_PILL_SPAN = 2 * 35


class PersonalityToggle(BigMultiToggle):
  def __init__(self, text: str, param: str, *, description: str = ""):
    super().__init__(text, list(PERSONALITY_LABELS), description=description)
    self._param = param
    self._params = Params()
    self.set_personality(self._params.get(self._param) or 0)

  def set_personality(self, personality: int) -> None:
    """enum 番号で表示を合わせる (車間ボタンで変わったときに toggles.py から呼ばれる)。"""
    if personality not in PERSONALITY_VALUES:
      personality = 1  # 範囲外は standard 扱い (selfdrived の丸めと同じ向き)
    self.set_value(PERSONALITY_LABELS[PERSONALITY_VALUES.index(personality)])

  def _handle_mouse_release(self, mouse_pos):
    super()._handle_mouse_release(mouse_pos)      # BigMultiToggle が表示上の次の段へ (下へ、最後は先頭へ)
    self._params.put(self._param, PERSONALITY_VALUES[PERSONALITY_LABELS.index(self.value)])

  def _draw_content(self, btn_y: float):
    BigButton._draw_content(self, btn_y)

    checked_idx = self._options.index(self.value)
    x = self._rect.x + self._rect.width - self._txt_enabled_toggle.width
    step = _PILL_SPAN / (len(self._options) - 1)
    for i in range(len(self._options)):
      self._draw_pill(x, btn_y + i * step, checked_idx == i)
