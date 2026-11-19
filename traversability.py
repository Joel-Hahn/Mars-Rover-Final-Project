"""Slope-based traversability and movement costs for a Mars rover.

Works directly with the Terrain object produced by terrain.py::load_case(),
which provides 2D ``elev`` (meters), ``slope_deg`` (degrees), and
``cell_size`` (meters; normally 2). This module does not load DTM data or
change the existing repository's main.py / terrain.py.

IMPORTANT -- RESEARCH PROTOTYPE, NOT FLIGHT-SAFE NAVIGATION
    - NASA-derived reference values: Perseverance mass ~1,025 kg; top speed
      152 m/hour. Base *effective* speed 110 m/hour is a model assumption.
    - The 15-degree safety threshold, speed multipliers, rolling resistance,
      drivetrain efficiency, and system power are configurable ASSUMPTIONS,
      not calibrated Perseverance telemetry.
    - DTM slope cannot reveal rocks, sand, wheel slip, hidden obstacles, or
      actual rover roll/pitch. Full rover footprint is NOT evaluated by default.
    - Energy is an illustrative electrical estimate, not a validated mission
      power budget. Negative/downhill mechanical work is not regenerated.

Units used throughout: meters, seconds, degrees, watts, joules, and Wh.
Rows increase southward; columns increase eastward, matching terrain.py.

Quick integration:
    from terrain import load_case
    from traversability import TraversabilityModel

    terrain = load_case(case)   # case is one dict from data/cases.json
    model = TraversabilityModel(terrain)
    tile = model.evaluate_tile(12, 15)
    move = model.evaluate_move((12, 15), (12, 16))
    summary = model.evaluate_path([(12, 15), (12, 16), (13, 16)])

The per-tile time/energy values are *reference* values for a level 2 m
cardinal crossing. Use evaluate_move() or evaluate_path() for real elevation
change, direction-specific travel time, and energy.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees, hypot, inf, isfinite
from typing import Mapping, Protocol, Sequence

import numpy as np


class TerrainLike(Protocol):
    """Minimal interface already implemented by terrain.Terrain."""

    elev: np.ndarray
    slope_deg: np.ndarray
    cell_size: float


@dataclass(frozen=True)
class RoverConfig:
    """Editable rover and safety assumptions.

    A move is blocked if a tile OR the center-to-center transition reaches
    its configured slope limit. Optional max_elevation_change_m is a cap on
    center-to-center elevation change, not a physical rock/step-height test.

    clearance_radius_cells=0 disables any full-rover-footprint approximation.
    Setting it to 1 requires the 3x3 neighborhood to be passable, which is
    conservative and should not be interpreted as a geometric rover model.
    """

    max_tile_slope_deg: float = 15.0
    max_transition_slope_deg: float = 15.0
    max_elevation_change_m: float | None = None
    clearance_radius_cells: int = 0
    base_speed_m_per_hour: float = 110.0
    slow_slope_deg: float = 5.0
    difficult_slope_deg: float = 10.0
    moderate_speed_factor: float = 0.75
    difficult_speed_factor: float = 0.40
    mass_kg: float = 1025.0
    mars_gravity_m_s2: float = 3.71
    default_rolling_resistance: float = 0.10
    drivetrain_efficiency: float = 0.60
    fixed_power_w: float = 100.0

    def __post_init__(self) -> None:
        positive = (
            "max_tile_slope_deg", "max_transition_slope_deg",
            "base_speed_m_per_hour", "difficult_slope_deg", "mass_kg",
            "mars_gravity_m_s2", "drivetrain_efficiency",
        )
        for field in positive:
            value = getattr(self, field)
            if not isfinite(value) or value <= 0:
                raise ValueError(f"{field} must be finite and > 0")
        if not (0 <= self.slow_slope_deg < self.difficult_slope_deg < 90):
            raise ValueError("Expected 0 <= slow_slope_deg < difficult_slope_deg < 90")
        for field in ("max_tile_slope_deg", "max_transition_slope_deg"):
            if getattr(self, field) >= 90:
                raise ValueError(f"{field} must be < 90 degrees")
        for field in ("moderate_speed_factor", "difficult_speed_factor"):
            value = getattr(self, field)
            if not isfinite(value) or not 0 < value <= 1:
                raise ValueError(f"{field} must be in (0, 1]")
        if self.drivetrain_efficiency > 1:
            raise ValueError("drivetrain_efficiency must not exceed 1")
        for field in ("default_rolling_resistance", "fixed_power_w"):
            value = getattr(self, field)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{field} must be finite and >= 0")
        if self.max_elevation_change_m is not None and (
            not isfinite(self.max_elevation_change_m)
            or self.max_elevation_change_m < 0
        ):
            raise ValueError("max_elevation_change_m must be finite and >= 0")
        if (not isinstance(self.clearance_radius_cells, int)
                or self.clearance_radius_cells < 0):
            raise ValueError("clearance_radius_cells must be an integer >= 0")


@dataclass(frozen=True)
class MaterialProperties:
    """Future surface-classification hook; not inferred from DTM elevation.

    Each material can change safety, speed and rolling resistance. For now,
    absent material data uses the 'unspecified' profile with neutral defaults.
    If a user supplies a material grid, an unrecognized label is BLOCKED.
    """

    passable: bool = True
    speed_multiplier: float = 1.0
    rolling_resistance: float | None = None
    max_slope_deg: float | None = None

    def __post_init__(self) -> None:
        if not isfinite(self.speed_multiplier) or self.speed_multiplier <= 0:
            raise ValueError("Material speed_multiplier must be > 0 and finite")
        if self.rolling_resistance is not None and (
            not isfinite(self.rolling_resistance) or self.rolling_resistance < 0
        ):
            raise ValueError("Material rolling_resistance must be >= 0 and finite")
        if self.max_slope_deg is not None and (
            not isfinite(self.max_slope_deg) or not 0 < self.max_slope_deg < 90
        ):
            raise ValueError("Material max_slope_deg must be in (0, 90)")


@dataclass(frozen=True)
class TileResult:
    """Tile classification; time/energy assume a *level* cardinal crossing."""

    row: int
    col: int
    passable: bool
    reason: str
    risk_level: str
    elevation_m: float
    slope_deg: float
    material: str
    reference_distance_m: float
    speed_m_per_hour: float
    time_s: float
    energy_j: float

    @property
    def energy_wh(self) -> float:
        return self.energy_j / 3600.0


@dataclass(frozen=True)
class MoveResult:
    """Direction-specific cost for one horizontal/vertical/diagonal step."""

    start: tuple[int, int]
    end: tuple[int, int]
    passable: bool
    reason: str
    horizontal_distance_m: float
    distance_m: float
    elevation_change_m: float
    transition_slope_deg: float
    speed_m_per_hour: float
    time_s: float
    energy_j: float

    @property
    def energy_wh(self) -> float:
        return self.energy_j / 3600.0


@dataclass(frozen=True)
class PathResult:
    """Summed costs for a path of adjacent (row, col) grid cells."""

    passable: bool
    reason: str
    distance_m: float
    time_s: float
    energy_j: float
    moves: tuple[MoveResult, ...]

    @property
    def energy_wh(self) -> float:
        return self.energy_j / 3600.0


class TraversabilityModel:
    """Evaluate slope, elevation transitions and simulated rover travel cost.

    Args:
        terrain: terrain.Terrain or any object with elev/slope_deg/cell_size.
        config: Optional RoverConfig for tunable assumptions.
        material_grid: Optional 2D array of material names, matching elev.
        materials: Map material name -> MaterialProperties. When omitted,
            only the default 'unspecified' surface is available.

    No import of terrain.py is required: this consumes its object directly.
    """

    def __init__(
        self,
        terrain: TerrainLike,
        config: RoverConfig | None = None,
        material_grid: np.ndarray | None = None,
        materials: Mapping[str, MaterialProperties] | None = None,
    ) -> None:
        self.terrain = terrain
        self.config = config or RoverConfig()
        self.elev = np.asarray(terrain.elev, dtype=float)
        self.slope_deg = np.asarray(terrain.slope_deg, dtype=float)
        self.cell_size = float(terrain.cell_size)
        if (self.elev.ndim != 2 or self.slope_deg.shape != self.elev.shape
                or not isfinite(self.cell_size) or self.cell_size <= 0):
            raise ValueError("Expected equal-sized 2D elevation/slope grids and positive cell_size")

        if material_grid is not None:
            self.material_grid = np.asarray(material_grid, dtype=object)
            if self.material_grid.shape != self.elev.shape:
                raise ValueError("material_grid must have same shape as terrain.elev")
        else:
            self.material_grid = None
        self.materials = {"unspecified": MaterialProperties()}
        if materials:
            self.materials.update(materials)

    def _validate_cell(self, row: int, col: int) -> None:
        if (not isinstance(row, (int, np.integer))
                or not isinstance(col, (int, np.integer))):
            raise TypeError("row and col must be integers")
        nrows, ncols = self.elev.shape
        if not (0 <= row < nrows and 0 <= col < ncols):
            raise IndexError(f"Cell ({row}, {col}) outside {nrows}x{ncols} grid")

    def _material_at(self, row: int, col: int) -> tuple[str, MaterialProperties | None]:
        name = ("unspecified" if self.material_grid is None
                else str(self.material_grid[row, col]))
        return name, self.materials.get(name)

    def _tile_status(self, row: int, col: int) -> tuple[bool, str]:
        elevation = float(self.elev[row, col])
        slope = float(self.slope_deg[row, col])
        if not isfinite(elevation) or not isfinite(slope) or slope < 0:
            return False, "Missing or invalid terrain elevation/slope"
        material, props = self._material_at(row, col)
        if props is None:
            return False, f"Unknown material {material!r}"
        if not props.passable:
            return False, f"Material {material!r} is blocked"
        limit = self.config.max_tile_slope_deg
        if props.max_slope_deg is not None:
            limit = min(limit, props.max_slope_deg)
        if slope >= limit:
            return False, f"Tile slope {slope:.2f} deg reaches limit {limit:.2f} deg"
        return True, "Passable under configured tile rules"

    def _clearance_status(self, row: int, col: int) -> tuple[bool, str]:
        radius = self.config.clearance_radius_cells
        if radius == 0:
            return True, ""
        rows, cols = self.elev.shape
        for rr in range(row - radius, row + radius + 1):
            for cc in range(col - radius, col + radius + 1):
                if not (0 <= rr < rows and 0 <= cc < cols):
                    return False, "Clearance neighborhood extends outside known terrain"
                ok, reason = self._tile_status(rr, cc)
                if not ok:
                    return False, f"Clearance blocked at ({rr}, {cc}): {reason}"
        return True, ""

    def _speed_factor(self, slope: float) -> float:
        if slope < self.config.slow_slope_deg:
            return 1.0
        if slope < self.config.difficult_slope_deg:
            return self.config.moderate_speed_factor
        return self.config.difficult_speed_factor

    def _speed_m_per_hour(self, slope: float, material: MaterialProperties) -> float:
        return (self.config.base_speed_m_per_hour
                * self._speed_factor(slope) * material.speed_multiplier)

    def _electrical_energy_j(
        self, travel_time_s: float, surface_distance_m: float,
        dz_m: float, horizontal_distance_m: float,
        rolling_resistance: float,
    ) -> float:
        cfg = self.config
        # Resistance force is Crr * normal force. For a uniform grade,
        # cos(theta) = horizontal / distance along the incline.
        cos_grade = horizontal_distance_m / surface_distance_m
        rolling_j = (rolling_resistance * cfg.mass_kg
                     * cfg.mars_gravity_m_s2 * cos_grade * surface_distance_m)
        gravity_j = cfg.mass_kg * cfg.mars_gravity_m_s2 * dz_m
        # Assume no regenerative braking or energy credit for going downhill.
        traction_electric_j = max(0.0, rolling_j + gravity_j) / cfg.drivetrain_efficiency
        return cfg.fixed_power_w * travel_time_s + traction_electric_j

    def evaluate_tile(self, row: int, col: int) -> TileResult:
        """Assess a tile and show nominal speed/time/energy over one flat 2 m crossing.

        Use evaluate_move() for a real 2D/3D transition and directional energy.
        A passable tile alone does not imply that entry from every neighbor is safe.
        """
        self._validate_cell(row, col)
        elevation, slope = float(self.elev[row, col]), float(self.slope_deg[row, col])
        name, material = self._material_at(row, col)
        ok, reason = self._tile_status(row, col)
        if ok:
            ok, clearance_reason = self._clearance_status(row, col)
            if not ok:
                reason = clearance_reason
        if not ok:
            return TileResult(row, col, False, reason, "blocked", elevation, slope,
                              name, self.cell_size, 0.0, inf, inf)

        if slope < self.config.slow_slope_deg:
            risk = "easy"
        elif slope < self.config.difficult_slope_deg:
            risk = "moderate"
        else:
            risk = "difficult"
        speed = self._speed_m_per_hour(slope, material)
        time_s = 3600.0 * self.cell_size / speed
        resistance = (material.rolling_resistance
                      if material.rolling_resistance is not None
                      else self.config.default_rolling_resistance)
        # Reference only: flat movement with 0 m height change, even for a
        # sloped tile. A tile alone does not specify the driving direction.
        energy = self._electrical_energy_j(time_s, self.cell_size, 0.0,
                                           self.cell_size, resistance)
        return TileResult(row, col, True, reason, risk, elevation, slope,
                          name, self.cell_size, speed, time_s, energy)

    def evaluate_move(
        self, start: tuple[int, int], end: tuple[int, int],
    ) -> MoveResult:
        """Evaluate a single 8-connected move, including elevation change.

        Source and destination tiles must be safe. Diagonal moves also require
        both adjacent orthogonal alternatives to be passable and have safe
        transitions, preventing corner-cutting through blocked cells.
        """
        if len(start) != 2 or len(end) != 2:
            raise ValueError("start and end must be (row, col)")
        sr, sc = start
        er, ec = end
        self._validate_cell(sr, sc)
        self._validate_cell(er, ec)
        dr, dc = er - sr, ec - sc
        if not (0 < max(abs(dr), abs(dc)) <= 1):
            raise ValueError("Only adjacent horizontal, vertical or diagonal moves are supported")
        horizontal = self.cell_size * hypot(dr, dc)
        dz = float(self.elev[er, ec] - self.elev[sr, sc])
        distance = hypot(horizontal, dz) if isfinite(dz) else float("nan")
        grade = degrees(atan2(abs(dz), horizontal)) if isfinite(dz) else float("nan")

        def blocked(reason: str) -> MoveResult:
            return MoveResult(tuple(start), tuple(end), False, reason, horizontal,
                              distance, dz, grade, 0.0, inf, inf)

        for row, col in (start, end):
            ok, reason = self._tile_status(row, col)
            if not ok:
                return blocked(f"Cell ({row}, {col}): {reason}")
            ok, reason = self._clearance_status(row, col)
            if not ok:
                return blocked(f"Cell ({row}, {col}): {reason}")

        if not isfinite(grade) or grade >= self.config.max_transition_slope_deg:
            return blocked(f"Transition slope {grade:.2f} deg is unsafe")
        if (self.config.max_elevation_change_m is not None
                and abs(dz) > self.config.max_elevation_change_m):
            return blocked("Center-to-center elevation change exceeds configured limit")

        if dr != 0 and dc != 0:
            # No slipping through blocked corners. Check two L-shaped routes.
            # Conservative policy: both alternatives must work. This only
            # evaluates the two cardinal transitions, never recurses diagonally.
            for middle in ((sr, ec), (er, sc)):
                middle_ok, reason = self._tile_status(*middle)
                if not middle_ok:
                    return blocked(f"Diagonal corner at {middle}: {reason}")
                clearance_ok, why = self._clearance_status(*middle)
                if not clearance_ok:
                    return blocked(f"Diagonal corner at {middle}: {why}")
                for a, b in ((start, middle), (middle, end)):
                    segment_dz = float(self.elev[b] - self.elev[a])
                    segment_grade = degrees(atan2(abs(segment_dz), self.cell_size))
                    if segment_grade >= self.config.max_transition_slope_deg:
                        return blocked("Diagonal corner has unsafe cardinal transition")
                    if (self.config.max_elevation_change_m is not None
                            and abs(segment_dz) > self.config.max_elevation_change_m):
                        return blocked("Diagonal corner exceeds elevation-change limit")

        _, src_material = self._material_at(sr, sc)
        _, dst_material = self._material_at(er, ec)
        # Conservative use of worse slope and slower material, averaged Crr.
        effective_slope = max(float(self.slope_deg[sr, sc]),
                              float(self.slope_deg[er, ec]), grade)
        speed = (self.config.base_speed_m_per_hour
                 * self._speed_factor(effective_slope)
                 * min(src_material.speed_multiplier, dst_material.speed_multiplier))
        time_s = 3600.0 * distance / speed
        def crr(props: MaterialProperties) -> float:
            return (self.config.default_rolling_resistance
                    if props.rolling_resistance is None
                    else props.rolling_resistance)
        rolling_resistance = (crr(src_material) + crr(dst_material)) / 2.0
        energy_j = self._electrical_energy_j(time_s, distance, dz, horizontal,
                                             rolling_resistance)
        return MoveResult(tuple(start), tuple(end), True, "Passable", horizontal,
                          distance, dz, grade, speed, time_s, energy_j)

    def evaluate_path(self, cells: Sequence[tuple[int, int]]) -> PathResult:
        """Sum time, distance and energy for adjacent cells in travel order."""
        if not cells:
            raise ValueError("Path must contain at least one cell")
        self._validate_cell(*cells[0])
        if len(cells) == 1:
            tile = self.evaluate_tile(*cells[0])
            return PathResult(tile.passable, tile.reason,
                              0.0 if tile.passable else inf,
                              0.0 if tile.passable else inf,
                              0.0 if tile.passable else inf, ())
        moves: list[MoveResult] = []
        for a, b in zip(cells, cells[1:]):
            step = self.evaluate_move(a, b)
            moves.append(step)
            if not step.passable:
                return PathResult(False, f"Blocked move {a} -> {b}: {step.reason}",
                                  inf, inf, inf, tuple(moves))
        return PathResult(True, "Passable", sum(m.distance_m for m in moves),
                          sum(m.time_s for m in moves),
                          sum(m.energy_j for m in moves), tuple(moves))

    def tile_maps(self) -> dict[str, np.ndarray]:
        """Return grid-shaped layers for a pathfinder/plotter.

        Time and energy here are reference flat/cardinal tile costs. Optimizers
        should call evaluate_move() for accurate directional edge weights.
        Blocked tiles have speed 0, time/energy inf.
        """
        shape = self.elev.shape
        passable = np.zeros(shape, dtype=bool)
        speed = np.zeros(shape, dtype=float)
        time_s = np.full(shape, inf, dtype=float)
        energy_j = np.full(shape, inf, dtype=float)
        for row, col in np.ndindex(shape):
            tile = self.evaluate_tile(row, col)
            passable[row, col] = tile.passable
            speed[row, col] = tile.speed_m_per_hour
            time_s[row, col] = tile.time_s
            energy_j[row, col] = tile.energy_j
        return {"passable": passable, "speed_m_per_hour": speed,
                "time_s": time_s, "energy_j": energy_j}


if __name__ == "__main__":
    # Standalone demonstration requiring only NumPy; no actual DTM needed.
    from types import SimpleNamespace

    # 3 rows x 4 columns of 2-meter cells; one steep and one no-data tile.
    demo_elev = np.array([[0.0, 0.1, 0.2, 0.2],
                          [0.0, 0.1, 0.2, np.nan],
                          [0.0, 0.1, 0.2, 0.2]])
    demo_slopes = np.array([[1.0, 4.0, 9.0, 16.0],
                            [1.0, 4.0, 9.0, np.nan],
                            [1.0, 4.0, 9.0, 4.0]])
    demo_terrain = SimpleNamespace(elev=demo_elev, slope_deg=demo_slopes, cell_size=2.0)
    model = TraversabilityModel(demo_terrain)
    print("Tile (0, 1):", model.evaluate_tile(0, 1))
    print("Move (0, 0) -> (0, 1):", model.evaluate_move((0, 0), (0, 1)))
    print("Path:", model.evaluate_path([(0, 0), (0, 1), (0, 2)]))
    print("Passability map (1=passable, 0=blocked):\n", model.tile_maps()["passable"].astype(int))
