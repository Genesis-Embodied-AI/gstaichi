from functools import partial
from typing import Any, TypeAlias
from weakref import ReferenceType

from gstaichi.lang import impl
from gstaichi.lang.impl import Program
from gstaichi.lang.kernel_arguments import ArgMetadata

from ._template_mapper_hotpath import _extract_arg

ArgsHash: TypeAlias = tuple[int, ...]
Key: TypeAlias = tuple[Any, ...]


def _destroy_callback(template_mapper_ref: ReferenceType["TemplateMapper"], ref: ReferenceType):
    maybe_template_mapper = template_mapper_ref()
    if maybe_template_mapper is not None:
        maybe_template_mapper._mapping_cache.clear()
        maybe_template_mapper._mapping_cache_tracker.clear()
        maybe_template_mapper._prog_weakref = None


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
        self.mapping: dict[Key, int] = {}
        self._mapping_cache: dict[ArgsHash, tuple[int, Key]] = {}
        self._mapping_cache_tracker: dict[ArgsHash, list[ReferenceType]] = {}
        self._prog_weakref: ReferenceType[Program] | None = None

    def extract(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> Key:
        return tuple(
            [
                _extract_arg(raise_on_templated_floats, arg, kernel_arg.annotation, kernel_arg.name)
                for arg, kernel_arg in zip(args, self.arguments)
            ]
        )

    def lookup(self, raise_on_templated_floats: bool, args: tuple[Any, ...]) -> tuple[int, Key]:
        if len(args) != self.num_args:
            raise TypeError(f"{self.num_args} argument(s) needed but {len(args)} provided.")

        # Keep track of taichi runtime to automatically clear cache if destroyed
        if self._prog_weakref is None:
            prog = impl.get_runtime().prog
            assert prog is not None
            self._prog_weakref = ReferenceType(prog, partial(_destroy_callback, ReferenceType(self)))
        else:
            # Since we already store a weak reference to taichi program, it is much faster to use it rather than
            # paying the overhead of calling pybind11 functions (~200ns vs 5ns).
            prog = self._prog_weakref()
        assert prog is not None

        mapping_cache_tracker: list[ReferenceType] | None = None
        args_hash: ArgsHash = tuple([id(arg) for arg in args])
        try:
            mapping_cache_tracker = self._mapping_cache_tracker[args_hash]
        except KeyError:
            pass
        if mapping_cache_tracker:
            return self._mapping_cache[args_hash]

        key = self.extract(raise_on_templated_floats, args)
        try:
            count = self.mapping[key]
        except KeyError:
            count = self.mapping[key] = len(self.mapping)

        mapping_cache_tracker_: list[ReferenceType] = []
        clear_callback = lambda ref: mapping_cache_tracker_.clear()
        try:
            mapping_cache_tracker_ += [ReferenceType(arg, clear_callback) for arg in args]
            self._mapping_cache_tracker[args_hash] = mapping_cache_tracker_
            self._mapping_cache[args_hash] = (count, key)
        except TypeError:
            pass

        return (count, key)
