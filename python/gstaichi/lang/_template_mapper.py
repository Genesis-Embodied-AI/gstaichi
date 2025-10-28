from typing import Any

from gstaichi.lang.kernel_arguments import ArgMetadata

from ._template_mapper_hotpath import _extract_arg


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
        print("template_slot_locations", template_slot_locations)
        self.arguments: list[ArgMetadata] = arguments
        self.num_args: int = len(arguments)
        self.template_slot_locations: list[int] = template_slot_locations
        self.mapping: dict[tuple[Any, ...], int] = {}
        self._fast_weak_map: dict = {}

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
        print("lookup", "args", args)

        fast_key = tuple([id(arg) for arg in args])
        print("fast_key", fast_key)
        if fast_key in self._fast_weak_map:
            print("fast key in fast weak map returning", self._fast_weak_map[fast_key])
            return self._fast_weak_map[fast_key]
        print("fast key not in map")
        key = self.extract(raise_on_templated_floats, args)
        print("key", key)
        try:
            res = self.mapping[key], key
            print('res', res)
            needs_grad = any([isinstance(arg, tuple) and len(arg) >= 3 and arg[2] for arg in args])
            print('needs_grad', needs_grad)
            if not needs_grad:
                print('storing in weak map key=', fast_key, 'res', res)
                self._fast_weak_map[fast_key] = res
            print('returning res', res)
            return res
        except KeyError:
            print("key not in self.mapping")
            count = len(self.mapping)
            print('count', count)
            print('setting self.mapping key', key, '=count', count)
            self.mapping[key] = count
            print('returning', count, key)
            return count, key
