"""Lane Centering の設定ウィジェット (mici)。

⚠⚠ **なぜ `BigParamControl` をそのまま使えないのか**: あれは `Params` 直叩きだが、この 4 つは
**openpilot の Params に置いていない**。`/data/params` に置くと `clearAll` のホワイトリスト
(= `libparams_c.so` に焼かれた表。release 配布では fork がヘッダに足しても載らない) から外れて
**manager 起動のたびに消される**ため (2026-08-26 に c4 で確認)。保存先は `/data/params_fork/d` で、
その読み書きは `lane_centering_params` が持っている。⇒ そこを経由する版をここに置く。

⚠ 新規ファイルにしているのは `widgets/button.py` が upstream 側のファイルで、追従のたびに
衝突しうるため。既存クラスの継承だけで済ませて button.py には手を入れない。

⚠ 書き込みは**成功したときだけ**表示を更新する。画面が ON なのに制御が OFF (あるいはその逆) に
なるのが一番危ない — 走行中に「切ったつもり」で切れていない状態を作らないこと。
"""
from collections.abc import Callable, Sequence

from openpilot.selfdrive.ui.mici.widgets.button import BigButton, BigMultiToggle, BigToggle
from openpilot.sunnypilot.selfdrive.controls.lib import lane_centering_params as lcp


class LaneCenteringToggle(BigToggle):
  """bool 1 個。`BigParamControl` の保存先違い版 (Params ではなくファイル)。"""

  def __init__(self, text: str, key: str):
    super().__init__(text, "")
    self._key = key
    self.refresh()

  def refresh(self) -> None:
    """保存先の現在値を画面に取り込む。SSH で直接書かれた場合もここで拾える。"""
    self.set_checked(lcp.read_bool(self._key))

  def _handle_mouse_release(self, mouse_pos):
    was = self._checked
    super()._handle_mouse_release(mouse_pos)      # BigToggle が _checked を反転
    if not lcp.write(self._key, self._checked):
      self.set_checked(was)                       # 書けなかったら見た目を戻す


class LaneCenteringChoice(BigMultiToggle):
  """float 1 個を選択肢から循環させる。

  ⚠ `BigMultiToggle` の pill は描かない。pill は `y += 35` で縦に積むので、ボタン高 180px に
  収まるのは 4 個まで (5 個目の下端が 206px で溢れる)。offset は 7 段欲しいので、選択状態は
  pill ではなく value 行の文字 ("0.10 m left") で示す。
  """

  def __init__(self, text: str, key: str, choices: Sequence[float], label_fn: Callable[[float], str]):
    self._key = key
    self._choices = tuple(float(c) for c in choices)
    # ⚠ 表示ラベルがそのまま BigMultiToggle の options になる (index() で引かれる) ので、
    #    重複するラベルを作らないこと。
    self._labels = [label_fn(c) for c in self._choices]
    assert len(set(self._labels)) == len(self._labels), f"{key}: 表示ラベルが重複している"
    super().__init__(text, self._labels)
    self.refresh()

  def refresh(self) -> None:
    """保存先の現在値に最も近い選択肢を選ぶ。

    ⚠ 「最も近い」で拾うのは、選択肢に無い値 (sunnylink や SSH で入れた 0.15 など) でも
    画面が何かしら妥当な位置を指すようにするため。⚠ ただし保存先は書き換えない —
    設定画面を開いただけで値が丸まるのは、こちらが触っていない設定を壊す挙動になる。
    """
    v = lcp.read_float(self._key)
    idx = min(range(len(self._choices)), key=lambda i: abs(self._choices[i] - v))
    BigMultiToggle.set_value(self, self._labels[idx])

  def _handle_mouse_release(self, mouse_pos):
    was = self.value
    super()._handle_mouse_release(mouse_pos)      # BigMultiToggle が次の選択肢へ
    value = self._choices[self._labels.index(self.value)]
    if not lcp.write(self._key, value):
      BigMultiToggle.set_value(self, was)

  def _draw_content(self, btn_y: float):
    # pill を描かない (docstring 参照)。BigMultiToggle を飛ばして BigButton の描画だけ使う
    BigButton._draw_content(self, btn_y)
