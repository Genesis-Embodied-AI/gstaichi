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
        self.arguments: list[ArgMetadata] = arguments
        self.num_args: int = len(arguments)
        self.template_slot_locations: list[int] = template_slot_locations
        self.mapping: dict[tuple[Any, ...], int] = {}

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

        key = self.extract(raise_on_templated_floats, args)
        try:
            return self.mapping[key], key
        except KeyError:
            count = len(self.mapping)
            self.mapping[key] = count
            return count, key
