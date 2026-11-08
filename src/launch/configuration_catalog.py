"""Bidirectional mapping between output classes and CUDA block dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator


@dataclass(frozen=True, order=True)
class Configuration:
    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self) -> None:
        if min(self.x, self.y, self.z) <= 0:
            raise ValueError("configuration dimensions must be positive")

    @property
    def threads(self) -> int:
        return self.x * self.y * self.z


class ConfigurationCatalog:
    def __init__(self, configurations: Iterable[Configuration]) -> None:
        self._configurations = tuple(configurations)
        if not self._configurations:
            raise ValueError("configuration catalog cannot be empty")
        if len(set(self._configurations)) != len(self._configurations):
            raise ValueError("configuration catalog contains duplicates")
        self._indices = {
            configuration: index
            for index, configuration in enumerate(self._configurations)
        }

    @classmethod
    def lscat(cls) -> "ConfigurationCatalog":
        """Sixteen one-dimensional actions used by the LS-CAT NLP task."""

        return cls(Configuration(size) for size in range(64, 1025, 64))

    @classmethod
    def lscat_all_shapes(cls) -> "ConfigurationCatalog":
        """All twenty shapes collected by the original LS-CAT dataset."""

        one_dimensional = [Configuration(size) for size in range(64, 1025, 64)]
        two_dimensional = [Configuration(size, size) for size in (8, 16, 24, 32)]
        return cls([*one_dimensional, *two_dimensional])

    def to_id(self, configuration: Configuration) -> int:
        return self._indices[configuration]

    def from_id(self, action_id: int) -> Configuration:
        return self._configurations[action_id]

    def __len__(self) -> int:
        return len(self._configurations)

    def __iter__(self) -> Iterator[Configuration]:
        return iter(self._configurations)

    def __getitem__(self, index: int) -> Configuration:
        return self._configurations[index]

