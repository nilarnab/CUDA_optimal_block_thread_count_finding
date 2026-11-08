import unittest

from launch.configuration_catalog import Configuration, ConfigurationCatalog


class ConfigurationCatalogTest(unittest.TestCase):
    def test_lscat_catalog_has_expected_bidirectional_mapping(self):
        catalog = ConfigurationCatalog.lscat()
        self.assertEqual(len(catalog), 16)
        self.assertEqual(catalog.from_id(0), Configuration(64, 1, 1))
        self.assertEqual(catalog.from_id(15), Configuration(1024, 1, 1))
        self.assertEqual(catalog.to_id(Configuration(256, 1, 1)), 3)

    def test_all_shape_catalog_retains_raw_twenty_actions(self):
        catalog = ConfigurationCatalog.lscat_all_shapes()
        self.assertEqual(len(catalog), 20)
        self.assertEqual(catalog.from_id(16), Configuration(8, 8, 1))
        self.assertEqual(catalog.to_id(Configuration(32, 32, 1)), 19)

    def test_rejects_duplicate_configurations(self):
        with self.assertRaises(ValueError):
            ConfigurationCatalog([Configuration(64), Configuration(64)])


if __name__ == "__main__":
    unittest.main()

