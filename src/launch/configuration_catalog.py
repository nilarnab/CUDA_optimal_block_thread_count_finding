from dataclasses import dataclass


@dataclass(frozen=True, order=True)
class Configuration:
    x: int
    y: int = 1
    z: int = 1

    def __post_init__(self):
        if min(self.x, self.y, self.z) <= 0:
            raise ValueError("configuration dimensions must be positive")

    @property
    def threads(self):
        return self.x * self.y * self.z


class ConfigurationCatalog:
    def __init__(self, configurations):
        self._configurations = tuple(configurations)
        self._indices = {
            configuration: index
            for index, configuration in enumerate(self._configurations)
        }

    @classmethod
    def lscat(cls):
        return cls(Configuration(size) for size in range(64, 1025, 64))

    @classmethod
    def lscat_all_shapes(cls):
        one_dimensional = [Configuration(size) for size in range(64, 1025, 64)]
        two_dimensional = [Configuration(size, size) for size in (8, 16, 24, 32)]
        return cls([*one_dimensional, *two_dimensional])

    def to_id(self, configuration):
        return self._indices[configuration]

    def from_id(self, action_id):
        return self._configurations[action_id]

    def __len__(self):
        return len(self._configurations)

    def __iter__(self):
        return iter(self._configurations)

    def __getitem__(self, index):
        return self._configurations[index]
