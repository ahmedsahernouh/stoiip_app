"""ZMAP node geometry and deterministic stock-tank oil initially in place."""

from dataclasses import dataclass
from io import BytesIO, TextIOWrapper
from math import isfinite
import re

import numpy as np


MAX_ZMAP_BYTES = 8 * 1024 * 1024
DEFAULT_MAX_NODES = 100_000
SQUARE_METRES_PER_ACRE = 4046.8564224
_NUMBER = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[EeDd][+-]?[0-9]+)?")
_INTEGER = re.compile(r"[+-]?[0-9]+")
_ALIGNMENT_TOLERANCE_M = 1e-6


def _grid_spacing(shape, xmin, xmax, ymin, ymax):
    if len(shape) != 2 or min(shape) < 2:
        raise ValueError("Grid dimensions must have at least two rows and two columns.")

    nrows, ncols = shape
    dx = (xmax - xmin) / (ncols - 1)
    dy = (ymax - ymin) / (nrows - 1)
    if not all(isfinite(value) for value in (xmin, xmax, ymin, ymax, dx, dy)) or min(dx, dy) <= 0:
        raise ValueError("Grid extents must be finite and strictly increasing in each axis.")
    return dx, dy


@dataclass(frozen=True, eq=False)
class Grid:
    """Rectangular grid with row zero at the north and column zero at the west."""

    values: np.ndarray
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def __post_init__(self):
        _grid_spacing(self.shape, self.xmin, self.xmax, self.ymin, self.ymax)

    @property
    def shape(self):
        return self.values.shape

    @property
    def x(self):
        return np.linspace(self.xmin, self.xmax, self.shape[1])

    @property
    def y(self):
        return np.linspace(self.ymax, self.ymin, self.shape[0])


def _float(token):
    if not _NUMBER.fullmatch(token):
        raise ValueError("Expected a numeric field.")
    value = float(token.replace("D", "E").replace("d", "e"))
    if not isfinite(value):
        raise ValueError("Numeric fields must be finite.")
    return value


def _integer(token):
    if not _INTEGER.fullmatch(token):
        raise ValueError("Expected an integer field.")
    return int(token)


def _records(stream):
    for number, line in enumerate(stream, 1):
        stripped = line.strip()
        if stripped and not stripped.startswith("!"):
            yield number, line.rstrip("\r\n")


def _data_line(line, width, start, nodes_per_line, remaining, parse):
    fixed = line[start:]
    if not line[:start].strip() and fixed and len(fixed) % width == 0:
        field_count = len(fixed) // width
        if field_count <= nodes_per_line:
            try:
                values = [
                    parse(fixed[offset:offset + width].strip())
                    for offset in range(0, min(field_count, remaining + 1) * width, width)
                ]
            except ValueError:
                pass
            else:
                if field_count > remaining:
                    raise ValueError("ZMAP sample count exceeds the declared dimensions.")
                return values

    fields = line.split(maxsplit=remaining)
    if len(fields) > remaining:
        raise ValueError("ZMAP sample count exceeds the declared dimensions.")
    return [parse(token) for token in fields]


def _read_grid(records, max_nodes):
    try:
        _, line = next(records)
        title = [field.strip() for field in line.split(",")]
        if title[0].upper() == "@GRID" and len(title) <= 2:
            nodes_per_line = _integer(title[1]) if len(title) == 2 else max_nodes
        elif len(title) == 3 and title[0].startswith("@") and title[1].upper() == "GRID":
            nodes_per_line = _integer(title[2])
        else:
            raise ValueError("Expected a ZMAP @GRID header.")
        if nodes_per_line < 1:
            raise ValueError("ZMAP line capacity must be positive.")

        _, line = next(records)
        width_text, numeric_null, text_null, decimals_text, start_text = (
            field.strip() for field in line.split(",")
        )
        width, decimals, first_column = map(_integer, (width_text, decimals_text, start_text))
        if width < 1 or not 0 <= decimals <= 308 or first_column < 1:
            raise ValueError("Invalid ZMAP field width, decimals or starting column.")
        null_value = _float(numeric_null) if numeric_null else None
        null_text = text_null if not numeric_null and text_null else None

        _, line = next(records)
        rows_text, cols_text, *extent_text = (field.strip() for field in line.split(","))
        if len(extent_text) != 4:
            raise ValueError("Expected four ZMAP extent coordinates.")
        shape = _integer(rows_text), _integer(cols_text)
        bounds = tuple(map(_float, extent_text))
        expected = shape[0] * shape[1]
        if expected > max_nodes:
            raise ValueError(f"ZMAP grid exceeds the {max_nodes:,}-node limit.")
        _grid_spacing(shape, *bounds)

        _, marker = next(records)
        if marker.strip() != "@":
            reserved = [_float(field.strip()) for field in marker.split(",")]
            if len(reserved) not in (3, 4) or any(reserved):
                raise ValueError("Unsupported ZMAP reserved header values.")
            _, marker = next(records)
        if marker.strip() != "@":
            raise ValueError("Expected an @ marker before ZMAP data.")
    except StopIteration as exc:
        raise ValueError("ZMAP header is incomplete.") from exc
    except ValueError as exc:
        raise ValueError(f"Invalid ZMAP header. {exc}") from exc

    scale = 10.0 ** -decimals

    def parse(token):
        if token == null_text:
            return np.nan
        value = _float(token)
        if value == null_value:
            return np.nan
        if decimals and _INTEGER.fullmatch(token):
            value *= scale
        return value

    values = np.empty(expected, dtype=float)
    count = 0
    for number, line in records:
        try:
            batch = _data_line(line, width, first_column - 1, nodes_per_line, expected - count, parse)
        except ValueError as exc:
            raise ValueError(f"Invalid ZMAP data on line {number}. {exc}") from exc
        values[count:count + len(batch)] = batch
        count += len(batch)

    if count != expected:
        raise ValueError(f"ZMAP sample count mismatch. Expected {expected:,} values, found {count:,}.")
    return Grid(values.reshape(shape[::-1]).T, *bounds)


def read_zmap(data: bytes, max_nodes: int = DEFAULT_MAX_NODES) -> Grid:
    """Read one unrotated ZMAP grid with its declared node extents."""
    if len(data) > MAX_ZMAP_BYTES:
        raise ValueError("ZMAP file exceeds the 8 MiB upload limit.")
    try:
        with TextIOWrapper(BytesIO(data), encoding="utf-8-sig") as stream:
            return _read_grid(_records(stream), max_nodes)
    except UnicodeDecodeError as exc:
        raise ValueError("ZMAP files must contain valid ASCII or UTF-8 text.") from exc


def require_aligned(reference: Grid, property: Grid) -> None:
    """Require matching node geometry with at most one micrometre of rounding."""
    if reference.shape != property.shape:
        raise ValueError("Property grid dimensions do not match the structure grid.")
    reference_bounds = reference.xmin, reference.xmax, reference.ymin, reference.ymax
    property_bounds = property.xmin, property.xmax, property.ymin, property.ymax
    reference_spacing = _grid_spacing(reference.shape, *reference_bounds)
    property_spacing = _grid_spacing(property.shape, *property_bounds)
    tolerance = min(_ALIGNMENT_TOLERANCE_M, 1e-6 * min(reference_spacing + property_spacing))
    if not np.allclose(
        reference_bounds + reference_spacing,
        property_bounds + property_spacing,
        rtol=0,
        atol=tolerance,
    ):
        raise ValueError("Property grid extents or increments do not match the structure grid.")


def node_area_acres(grid: Grid) -> np.ndarray:
    """Convert XY metres to acres using rectangular trapezoid node weights."""
    dx, dy = _grid_spacing(grid.shape, grid.xmin, grid.xmax, grid.ymin, grid.ymax)
    area_per_interior_node = dx / SQUARE_METRES_PER_ACRE * dy
    if not isfinite(area_per_interior_node) or area_per_interior_node <= 0:
        raise ValueError("Grid node area is outside the supported numeric range.")

    area = np.full(grid.shape, area_per_interior_node)
    area[[0, -1], :] *= 0.5
    area[:, [0, -1]] *= 0.5
    return area


def stoiip_stb(area_acres, thickness_ft, phi, so, ntg, bo):
    """Calculate stock-tank barrels using acres, feet and fractional properties."""
    return 7758.0 * area_acres * thickness_ft * phi * so * ntg / bo


def _property_array(values, shape, label, fraction):
    raw = np.asarray(values)
    if np.iscomplexobj(raw):
        raise ValueError(f"{label} must contain real numbers.")
    array = np.asarray(raw, dtype=float)
    if array.ndim == 0:
        if not isfinite(float(array)):
            raise ValueError(f"{label} must be finite when entered as a scalar.")
        array = np.broadcast_to(array, shape)
    if array.shape != shape:
        raise ValueError(f"{label} grid must match the structure grid dimensions.")

    valid = ~np.isnan(array)
    values = array[valid]
    if fraction:
        if not np.all((values >= 0) & (values <= 1)):
            raise ValueError(f"{label} must be a finite fraction between 0 and 1.")
    elif not np.all(np.isfinite(values) & (values > 0)):
        raise ValueError(f"{label} must be finite and greater than zero.")
    return array, valid


def weighted_volume(
    grid: Grid, thickness, phi, so, ntg, bo, contact=None, depth_positive_down=True,
) -> tuple[float, float, np.ndarray]:
    """Return STOIIP in STB, active acres and the masked per-node STB contribution."""
    if np.iscomplexobj(grid.values) or np.any(np.isinf(grid.values)):
        raise ValueError("Structure grid must contain finite real numbers or null values.")
    active = np.isfinite(grid.values)
    if contact is not None:
        if not isfinite(contact):
            raise ValueError("The oil-water contact must be finite.")
        if depth_positive_down:
            active &= grid.values <= contact
        else:
            active &= grid.values >= contact

    properties = (
        (thickness, "Thickness", False),
        (phi, "Porosity", True),
        (so, "Oil saturation", True),
        (ntg, "Net-to-gross", True),
        (bo, "Oil formation volume factor", False),
    )
    property_arrays = []
    for values, label, fraction in properties:
        array, valid = _property_array(values, grid.shape, label, fraction)
        active &= valid
        property_arrays.append(array)

    area = node_area_acres(grid)[active]
    node_volume = np.full(grid.shape, np.nan)
    with np.errstate(over="ignore", invalid="ignore"):
        contributions = stoiip_stb(area, *(array[active] for array in property_arrays))
        total_volume = float(contributions.sum())
        active_area = float(area.sum())
    if not isfinite(total_volume) or not isfinite(active_area):
        raise ValueError("The calculated area or STOIIP exceeds the supported numeric range.")
    node_volume[active] = contributions
    return total_volume, active_area, node_volume
