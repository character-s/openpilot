"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

import io
import struct
import pickle
import inspect
import importlib
import enum


def _pad_args(func, args, kwargs):
  try:
    sig = inspect.signature(func)
  except Exception:
    return args, kwargs
  params = list(sig.parameters.values())
  if inspect.isfunction(func) and params and params[0].name in ('cls', 'self'):
    params = params[1:]

  new_args = list(args)
  has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)
  if len(new_args) > len(params) and not has_varargs:
    new_args = new_args[:len(params)]

  for i in range(len(new_args), len(params)):
    param = params[i]
    if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
      continue
    val = param.default if param.default is not inspect.Parameter.empty else None
    new_args.append(val)
  return new_args, kwargs


# GS: Ops は pkl に **値 (序数)** で焼かれるので、enum に項目が挿入されたビルドで焼いた pkl は
# 値がズレて別の Ops として復元される。comma 本家 (tinygrad 9d0446a4) で焼かれた CTMv3 を
# こちらで読むと、値 56 が本家の CONST ではなく ENDIF、67 が RESHAPE ではなく MSTACK になり、
# JIT の入力照合が `args mismatch in JIT` で必ず落ちた (09-20 に実車で踏んだ)。
# ⚠ 差は「**こちらにだけある 2 個**」だけで、本家にだけある Ops は無い (突き合わせ済み) ので、
#   こちらの一覧からこの 2 個を除けば本家の並びがそのまま再現できる。名前で読み替えるのが正。
# ⚠ 増やすときは本家の enum と突き合わせてから。適当に足すと全モデルの復元が壊れる。
_OPS_ONLY_LOCAL = ("RETURNED", "CONTIGUOUS")


def _upstream_enum_member(enum_class, value: int):
  """本家 tinygrad の序数として value を解釈し、こちらの同名メンバを返す。"""
  members = [m for m in enum_class if m.name not in _OPS_ONLY_LOCAL]
  index = value - 1  # FastEnum の値は 1 始まり
  if 0 <= index < len(members):
    return members[index]
  return enum_class(value)


def _enum_factory(enum_class, upstream_ops: bool = False):
  # ⚠ 読み替えるのは Ops だけ。他の enum まで触ると復元が壊れる。
  remap = upstream_ops and enum_class.__name__ == "Ops"

  def factory(*args, **kwargs):
    try:
      if remap and len(args) == 1 and not kwargs and isinstance(args[0], int):
        return _upstream_enum_member(enum_class, args[0])
      return enum_class(*args, **kwargs)
    # OptOps and UOp objects in the .pkl are left over from the compilation phase,
    # reassignment does nothing because they aren't tied to the execution graph
    # It never executes or evaluates the UOp nodes again.
    except ValueError:
      return list(enum_class)[0]
  factory.__name__ = enum_class.__name__
  factory.__module__ = enum_class.__module__
  return factory


def _dynamic_factory(real_class, upstream_ops: bool = False):
  if isinstance(real_class, type) and issubclass(real_class, enum.Enum):
    return _enum_factory(real_class, upstream_ops)

  def factory(*args, **kwargs):
    try:
      return real_class(*args, **kwargs)
    except TypeError:
      new_args, new_kwargs = _pad_args(real_class, args, kwargs)
      return real_class(*new_args, **new_kwargs)

  class DynamicMeta(type(real_class)):
    def __call__(cls, *args, **kwargs):
      return factory(*args, **kwargs)

  class DynamicProxy(real_class, metaclass=DynamicMeta):
    __slots__ = ()

    def __new__(cls, *args, **kwargs):
      return factory(*args, **kwargs)

  DynamicProxy.__name__ = real_class.__name__
  DynamicProxy.__module__ = real_class.__module__
  return DynamicProxy


class DynamicTinygradUnpickler(pickle.Unpickler):
  def __init__(self, *args, upstream_ops: bool = False, **kwargs):
    super().__init__(*args, **kwargs)
    self.upstream_ops = upstream_ops

  def find_class(self, module, name):
    if module == "tinygrad.ops":
      try:
        importlib.import_module("tinygrad.uops")
        module = "tinygrad.uops"
      except ImportError:
        pass
    real_class = getattr(importlib.import_module(module), name)
    if module.startswith("tinygrad"):
      return _dynamic_factory(real_class, self.upstream_ops)
    return real_class


def load_oob(f, upstream_ops: bool = False):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      f.readinto(pb)
      yield pb
  return DynamicTinygradUnpickler(io.BytesIO(opcodes), buffers=buffers(), upstream_ops=upstream_ops).load()
