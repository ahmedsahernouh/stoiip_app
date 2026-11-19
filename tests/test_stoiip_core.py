import unittest

import numpy as np

from stoiip_core import (
    MAX_ZMAP_BYTES,
    Grid,
    node_area_acres,
    read_zmap,
    require_aligned,
    stoiip_stb,
    weighted_volume,
)


def zmap(payload, dimensions="2,3,100,300,200,300", layout="15,-9999,,0,1"):
    return (
        f"! example\n@Test grid, GRID, 5\n{layout}\n\n{dimensions}\n0,0,0\n@\n"
        f"{payload}\n"
    ).encode("utf-8")


class ZmapTest(unittest.TestCase):
    def test_column_major_order_and_numeric_null(self):
        grid = read_zmap(zmap(" 1D0 4\n! comment\n\n2 5\n3 -9.999d3"))
        np.testing.assert_allclose(grid.values, [[1, 2, 3], [4, 5, np.nan]])
        np.testing.assert_array_equal(grid.x, [100, 200, 300])
        np.testing.assert_array_equal(grid.y, [300, 200])

    def test_touching_fixed_width_fields_and_implicit_decimals(self):
        grid = read_zmap(zmap(
            "  1234-567\n  3456NULL",
            dimensions="2,2,0,1,0,1",
            layout="4,,NULL,2,3",
        ))
        np.testing.assert_allclose(grid.values, [[12.34, 34.56], [-5.67, np.nan]])

    def test_fixed_width_can_include_trailing_field_padding(self):
        data = zmap("1234-5  \n3 4", "2,2,0,1,0,1", "4,-9999,,0,1")
        np.testing.assert_array_equal(read_zmap(data).values, [[1234, 3], [-5, 4]])

    def test_invalid_counts_and_tokens(self):
        for payload in ("1 2 3 4 5", "1 2 3 4 5 6 7", "1 2 3 TYPO 5 6", "1 2 3 4 nan 6"):
            with self.subTest(payload=payload), self.assertRaisesRegex(ValueError, "ZMAP (sample count|data)"):
                read_zmap(zmap(payload))

    def test_extra_fixed_width_values_rejected(self):
        data = zmap("1\n2\n3\n4567", "2,2,0,1,0,1", "2,-9999,,0,1")
        with self.assertRaisesRegex(ValueError, "sample count"):
            read_zmap(data)

    def test_invalid_geometry(self):
        for dimensions in ("1,2,0,1,0,1", "2,2,0,0,0,1", "2,2,1,0,0,1", "2,2,0,1e309,0,1"):
            with self.subTest(dimensions=dimensions), self.assertRaisesRegex(ValueError, "Invalid ZMAP header"):
                read_zmap(zmap("", dimensions))

    def test_limits_are_checked_before_payload(self):
        with self.assertRaisesRegex(ValueError, "100,000-node limit"):
            read_zmap(zmap("broken", "5000000000000000,5000000000000000,0,1,0,1"))
        with self.assertRaisesRegex(ValueError, "4-node limit"):
            read_zmap(zmap("broken"), max_nodes=4)
        with self.assertRaisesRegex(ValueError, "upload limit"):
            read_zmap(b" " * (MAX_ZMAP_BYTES + 1))

    def test_header_marker_required(self):
        with self.assertRaisesRegex(ValueError, "Expected an @ marker"):
            read_zmap(zmap("1 2 3 4 5 6").replace(b"\n@\n", b"\nwrong\n"))

    def test_alignment_requires_identical_shape_and_coordinate_registration(self):
        grid = Grid(np.ones((2, 3)), 700000, 700100, 3000000, 3000200)
        require_aligned(grid, Grid(np.ones((2, 3)), 700000.00000005, 700100, 3000000, 3000200))
        with self.assertRaisesRegex(ValueError, "extents or increments"):
            require_aligned(grid, Grid(np.ones((2, 3)), 700000.001, 700100.001, 3000000, 3000200))
        with self.assertRaisesRegex(ValueError, "dimensions"):
            require_aligned(grid, Grid(np.ones((3, 3)), 700000, 700100, 3000000, 3000200))


class VolumeTest(unittest.TestCase):
    def test_manual_known_answer(self):
        self.assertAlmostEqual(stoiip_stb(100, 50, 0.2, 0.8, 0.5, 1.25), 2_482_560)

    def test_full_rectangle_trapezoid_node_weights(self):
        grid = Grid(np.ones((3, 3)), 0, 200, 0, 100)
        area = 20_000 / 4046.8564224
        expected_fractions = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 16
        np.testing.assert_allclose(node_area_acres(grid), area * expected_fractions)
        volume, active_area, contributions = weighted_volume(grid, 50, 0.2, 0.8, 0.5, 1.25)
        expected_volume = 24_825.6 * area
        self.assertAlmostEqual(active_area, area)
        self.assertAlmostEqual(volume, expected_volume)
        np.testing.assert_allclose(contributions, expected_volume * expected_fractions)

    def test_contact_signs_and_equality(self):
        structure = np.array([[100.0, 120.0], [90.0, 110.0]])
        grid = Grid(structure, 0, 100, 0, 100)
        elevation = Grid(-structure, 0, 100, 0, 100)
        args = (50, 0.2, 0.8, 0.5, 1.25)

        volume, area, nodes = weighted_volume(grid, *args, contact=110)
        other_volume, other_area, other_nodes = weighted_volume(
            elevation, *args, contact=-110, depth_positive_down=False,
        )
        self.assertAlmostEqual(area, 7_500 / 4046.8564224)
        self.assertAlmostEqual(volume, area * 24_825.6)
        self.assertEqual((volume, area), (other_volume, other_area))
        np.testing.assert_array_equal(np.isfinite(nodes), [[True, False], [True, True]])
        np.testing.assert_allclose(nodes, other_nodes)

    def test_excludes_null_structure_and_property_with_array_inputs(self):
        grid = Grid(np.array([[1.0, np.nan], [1.0, 1.0]]), 0, 100, 0, 100)
        phi = np.array([[0.25, 0.5], [np.nan, 0.0]])
        bo = np.array([[1.25, 1.25], [1.25, 1.25]])
        volume, area, nodes = weighted_volume(grid, 10, phi, 0.8, 0.5, bo)
        area_per_node = 2_500 / 4046.8564224
        self.assertAlmostEqual(area, 2 * area_per_node)
        self.assertAlmostEqual(volume, 6_206.4 * area_per_node)
        np.testing.assert_allclose(nodes, [[volume, np.nan], [np.nan, 0.0]])

    def test_no_active_nodes_returns_zero_and_a_masked_map(self):
        grid = Grid(np.ones((2, 2)), 0, 1, 0, 1)
        volume, area, nodes = weighted_volume(grid, 10, 0.2, 0.8, 0.5, 1.25, contact=0)
        self.assertEqual((volume, area), (0, 0))
        self.assertTrue(np.isnan(nodes).all())

    def test_physical_bounds_and_grid_shape(self):
        grid = Grid(np.ones((2, 2)), 0, 1, 0, 1)
        defaults = {"thickness": 10, "phi": 0.2, "so": 0.8, "ntg": 0.5, "bo": 1.25}
        for changes in (
            {"thickness": 0}, {"thickness": -1}, {"phi": -0.01}, {"so": 1.01},
            {"ntg": np.inf}, {"bo": 0}, {"bo": np.nan}, {"bo": np.ones((1, 2))},
            {"contact": np.inf}, {"phi": np.array([[np.inf, 1.0], [1.0, 1.0]])},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                weighted_volume(grid, **(defaults | changes))


if __name__ == "__main__":
    unittest.main()
