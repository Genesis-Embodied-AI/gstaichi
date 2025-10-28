from typing import Any
import weakref
from dataclasses import dataclass
import numpy as np

from gstaichi.lang.kernel_arguments import ArgMetadata

from ._template_mapper_hotpath import _extract_arg


@dataclass
class VerificationInfo:
    is_weak_ref: bool
    ref: weakref.ReferenceType | None = None
    value: int | float | bool | None = None

    def check(self, arg):
        if self.is_weak_ref:
            assert self.ref is not None
            return self.ref() is arg
        return arg == self.value


primitive_types = {float, int, bool, np.float64, np.int64, np.float32, np.int32, np.bool_}


class TemplateMapper:
    """
    This should probably be renamed to sometihng like FeatureMapper, or
    FeatureExtractor, since:
    - it's not specific to templates
    - it extracts what are later called 'features', for example for ndarray this includes:
        - element type
        - number dimensions
        - needs grad (or not)
    - these are returned as a heterogeneous tuple, whose contents depends on the type
    """

    def __init__(self, arguments: list[ArgMetadata], template_slot_locations: list[int]) -> None:
        self.arguments: list[ArgMetadata] = arguments
        self.num_args: int = len(arguments)
        self.template_slot_locations: list[int] = template_slot_locations
        self.mapping: dict[tuple[Any, ...], int] = {}
        self._fast_weak_map: dict[tuple[int, ...], tuple[int, tuple[Any, ...]]] = {}  # dict from tuple of ids of objects to the lookup result
        # dict from id to verification info, so we can check the id still refers to the same object
        self._verif_info_by_id: dict[int, VerificationInfo] = {}

    def extract(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(
            [
                _extract_arg(raise_on_templated_floats, arg, kernel_arg.annotation, kernel_arg.name)
                for arg, kernel_arg in zip(args, self.arguments)
            ]
        )

    def lookup(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[int, tuple[Any, ...]]:
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        fast_key = tuple([id(arg) for arg in args])
        if fast_key in self._fast_weak_map:
            # check the ids still match the objects
            _valid = True
            for _id, arg in zip(fast_key, args):
                verif_info = self._verif_info_by_id[_id]
                if not verif_info.check(arg):
                    _valid = False
                    del self._fast_weak_map[fast_key]
                    break
            if _valid:
                return self._fast_weak_map[fast_key]
        key = self.extract(raise_on_templated_floats, args)
        try:
            res = self.mapping[key], key
        except KeyError:
            count = len(self.mapping)
            self.mapping[key] = count
            res = count, key
        needs_grad = any([isinstance(arg, tuple) and len(arg) >= 3 and arg[2] for arg in args])
        if not needs_grad:
            ok_to_cache = True
            if any(isinstance(arg, tuple) for arg in args):
                ok_to_cache = False
            if ok_to_cache:
                for _id, arg in zip(fast_key, args):
                    arg_type = type(arg)
                    if arg_type in primitive_types:
                        verif_info = VerificationInfo(
                            is_weak_ref=False,
                            value=arg
                        )
                    else:
                        verif_info = VerificationInfo(
                            is_weak_ref=True,
                            ref=weakref.ref(arg)
                        )
                    self._verif_info_by_id[_id] = verif_info
                self._fast_weak_map[fast_key] = res
        return res
