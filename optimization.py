"""Optimal time and energy routes through the Mars rover traversability grid.

Part of the Mars Rover Final Project. This file uses the existing interfaces:

    terrain.py         -> Terrain.elev, Terrain.slope_deg, Terrain.cell_size,
                          Terrain.lonlat_to_cell(), Terrain.cell_to_lonlat(),
                          and load_case(case).
    traversability.py  -> TraversabilityModel.evaluate_tile/evaluate_move.

The first two searches are INDEPENDENT and DO NOT model battery charge,
charging stops, or a maximum remaining battery capacity:

    - Fastest route: minimizes the sum of movement times (seconds).
    - Lowest-energy route: minimizes the sum of movement energies (joules).

Both use 8-connected A* on the 2 x 2 m cell grid (or optionally four-connected).
The heuristics are conservative lower bounds for the current traversability
model; therefore, the routes are minimum-cost under *that model*. Dangerous
cells/edges are rejected by traversability.py, including unsafe diagonal
corner cuts. Energy need not be symmetric: climbing can cost more than descending.

IMPORTANT: This is a research/simulation model, NOT a flight-safe navigation
system. Thresholds and energy coefficients in traversability.py are not
calibrated Perseverance flight telemetry. Do not interpret the routes as
physically proven safe or their energy/time estimates as measured results.

Typical use:

    from terrain import load_case
    from traversability import TraversabilityModel
    from optimization import PathOptimizer

    terrain = load_case(case)  # a case from data/cases.json
    start = terrain.lonlat_to_cell(case['start']['lon'], case['start']['lat'])
    goal = terrain.lonlat_to_cell(case['end']['lon'], case['end']['lat'])
    result = PathOptimizer(TraversabilityModel(terrain)).optimize(start, goal)

    if result.fastest.found:
        print(result.fastest.time_s, result.fastest.energy_wh)
        print(result.fastest.cells)  # route (row, col) cell centres for plotting
        print(route_lonlat(terrain, result.fastest))

The module can also be run without real terrain files: python optimization.py

FUTURE BATTERY / RECHARGING EXTENSION:
A charging-aware optimizer requires *new search states*, such as
(row, col, discretized_state_of_charge), plus battery capacity, initial charge,
charging sites, charge rate and charger availability. Adding a battery limit
to the two existing route searches would change their meaning; deliberately
keep that future search separate (e.g., PathOptimizer.find_charging_route()).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from heapq import heappop, heappush
from math import hypot, inf, isfinite
from numbers import Integral
from typing import Mapping, Sequence

from traversability import (
    MaterialProperties,
    MoveResult,
    RoverConfig,
    TraversabilityModel,
)

Cell = tuple[int, int]

# Cardinal steps first; diagonals optionally included. The ordering breaks
# ties reproducibly (though multiple mathematically equal paths may exist).
_CARDINAL: tuple[Cell, ...] = ((-1, 0), (0, -1), (0, 1), (1, 0))
_DIAGONAL: tuple[Cell, ...] = ((-1, -1), (-1, 1), (1, -1), (1, 1))


@dataclass(frozen=True)
class OptimizedRoute:
    """One requested route and its full traversal statistics.

    No-path results have found=False, cells/moves empty and cost metrics=inf.
    Units: distance_m and horizontal_distance_m in m; time_s in seconds;
    energy_j in joules; ascent/descent in metres. Each move has its own
    time, energy, slope, speed, and elevation change, as returned by
    traversability.py.
    """

    objective: str
    found: bool
    reason: str
    start: Cell
    goal: Cell
    cells: tuple[Cell, ...]
    moves: tuple[MoveResult, ...]
    distance_m: float
    horizontal_distance_m: float
    time_s: float
    energy_j: float
    ascent_m: float
    descent_m: float
    expanded_nodes: int

    @property
    def energy_wh(self) -> float:
        return self.energy_j / 3600.0

    @property
    def number_of_moves(self) -> int:
        return len(self.moves)

    @property
    def average_speed_m_per_hour(self) -> float:
        return 3600.0 * self.distance_m / self.time_s if self.time_s > 0 and isfinite(self.time_s) else 0.0

    @property
    def max_transition_slope_deg(self) -> float:
        return max((step.transition_slope_deg for step in self.moves), default=0.0)

    def to_dict(self, *, include_moves: bool = False) -> dict:
        """JSON-compatible summary for the UI or a project report.

        Missing routes have None rather than Infinity in numeric fields.
        Row/column pairs are returned as [row, col] lists for JSON.
        """
        value = (lambda n: n if isfinite(n) else None)
        data = {
            "objective": self.objective,
            "found": self.found,
            "reason": self.reason,
            "start": list(self.start),
            "goal": list(self.goal),
            "cells": [list(cell) for cell in self.cells],
            "number_of_moves": self.number_of_moves,
            "distance_m": value(self.distance_m),
            "horizontal_distance_m": value(self.horizontal_distance_m),
            "time_s": value(self.time_s),
            "time_min": value(self.time_s / 60.0),
            "energy_j": value(self.energy_j),
            "energy_wh": value(self.energy_wh),
            "average_speed_m_per_hour": self.average_speed_m_per_hour,
            "ascent_m": value(self.ascent_m),
            "descent_m": value(self.descent_m),
            "max_transition_slope_deg": self.max_transition_slope_deg,
            "expanded_nodes": self.expanded_nodes,
        }
        if include_moves:
            data["moves"] = [
                {
                    "start": list(m.start),
                    "end": list(m.end),
                    "distance_m": m.distance_m,
                    "horizontal_distance_m": m.horizontal_distance_m,
                    "elevation_change_m": m.elevation_change_m,
                    "transition_slope_deg": m.transition_slope_deg,
                    "speed_m_per_hour": m.speed_m_per_hour,
                    "time_s": m.time_s,
                    "energy_j": m.energy_j,
                    "energy_wh": m.energy_wh,
                }
                for m in self.moves
            ]
        return data


@dataclass(frozen=True)
class OptimizationResults:
    """Two independent routes, optimized without battery recharging."""

    fastest: OptimizedRoute
    lowest_energy: OptimizedRoute

    def to_dict(self, *, include_moves: bool = False) -> dict:
        return {
            "fastest": self.fastest.to_dict(include_moves=include_moves),
            "lowest_energy": self.lowest_energy.to_dict(include_moves=include_moves),
        }


class PathOptimizer:
    """A* shortest-path optimizer using traversability.py movement costs.

    Args:
        model: Existing TraversabilityModel, including optional surface data.
        allow_diagonal: Search 8 neighbours if True, otherwise 4.
        max_expanded_nodes: Optional search limit, helpful for very large grids.
            A limited search can return found=False without proving no path.

    Grid indexing is (row, col), NOT (x, y) and NOT (longitude, latitude).
    Searches can revisit cells conceptually; no capacity/recharge state is used.
    All costs are nonnegative by the model, as required by A*.
    """

    def __init__(
        self,
        model: TraversabilityModel,
        *,
        allow_diagonal: bool = True,
        max_expanded_nodes: int | None = None,
    ) -> None:
        if not isinstance(model, TraversabilityModel):
            raise TypeError("model must be a TraversabilityModel")
        if max_expanded_nodes is not None and (
            not isinstance(max_expanded_nodes, Integral)
            or isinstance(max_expanded_nodes, bool)
            or max_expanded_nodes <= 0
        ):
            raise ValueError("max_expanded_nodes must be a positive integer or None")
        self.model = model
        self.allow_diagonal = bool(allow_diagonal)
        self.max_expanded_nodes = max_expanded_nodes
        self._offsets = _CARDINAL + (_DIAGONAL if self.allow_diagonal else ())
        self._rows, self._cols = model.elev.shape
        # Per-instance cache: avoids retaining unrelated models in a global
        # @lru_cache on a bound method, while bounding RAM on large maps.
        self._move = lru_cache(maxsize=65536)(model.evaluate_move)

        # Upper bound on travel speed for the A* time lower bound. A custom
        # material can increase speed, so don't assume max speed=base speed.
        max_material_speed_factor = max(
            (m.speed_multiplier for m in model.materials.values() if m.passable),
            default=1.0,
        )
        self._max_speed_m_per_hour = (
            model.config.base_speed_m_per_hour * max_material_speed_factor
        )
        # Lower bound on rolling resistance for energy heuristics. Actual
        # edge resistance is the average of its two material values.
        self._min_rolling_resistance = min(
            (
                model.config.default_rolling_resistance
                if m.rolling_resistance is None else m.rolling_resistance
                for m in model.materials.values() if m.passable
            ),
            default=0.0,
        )

    @staticmethod
    def _cell(cell: Sequence[int]) -> Cell:
        if len(cell) != 2 or any(
            isinstance(v, bool) or not isinstance(v, Integral) for v in cell
        ):
            raise TypeError("Grid position must contain exactly two integers: (row, col)")
        return (int(cell[0]), int(cell[1]))

    def _neighbors(self, current: Cell):
        row, col = current
        for dr, dc in self._offsets:
            rr, cc = row + dr, col + dc
            if 0 <= rr < self._rows and 0 <= cc < self._cols:
                yield (rr, cc)

    def _heuristic(self, cell: Cell, goal: Cell, objective: str) -> float:
        # Direct horizontal distance (in metres) is a lower bound on the
        # travelled distance of *any* path, including diagonal detours.
        direct_m = self.model.cell_size * hypot(goal[0] - cell[0], goal[1] - cell[1])
        min_time_s = 3600.0 * direct_m / self._max_speed_m_per_hour
        if objective == "time":
            return min_time_s

        # Energy: time-dependent power plus a conservative mechanical lower
        # bound from minimum rolling resistance and net elevation gain.
        # Summing actual max(0, rolling + gravitational) per move cannot cost
        # less than max(0, lower-bound rolling + net gravitational change).
        cfg = self.model.config
        gain_m = float(self.model.elev[goal] - self.model.elev[cell])
        traction_lb = max(
            0.0,
            cfg.mass_kg * cfg.mars_gravity_m_s2
            * (self._min_rolling_resistance * direct_m + gain_m),
        ) / cfg.drivetrain_efficiency
        return cfg.fixed_power_w * min_time_s + traction_lb

    @staticmethod
    def _not_found(
        objective: str, start: Cell, goal: Cell, reason: str, expanded: int = 0
    ) -> OptimizedRoute:
        return OptimizedRoute(
            objective, False, reason, start, goal, (), (),
            inf, inf, inf, inf, inf, inf, expanded,
        )

    @staticmethod
    def _assemble(
        objective: str,
        start: Cell,
        goal: Cell,
        previous: Mapping[Cell, tuple[Cell, MoveResult]],
        expanded: int,
    ) -> OptimizedRoute:
        cells = [goal]
        steps: list[MoveResult] = []
        current = goal
        while current != start:
            parent, step = previous[current]
            steps.append(step)
            current = parent
            cells.append(current)
        cells.reverse()
        steps.reverse()
        return OptimizedRoute(
            objective=objective,
            found=True,
            reason="Route found (no charging stops modeled)",
            start=start,
            goal=goal,
            cells=tuple(cells),
            moves=tuple(steps),
            distance_m=sum(m.distance_m for m in steps),
            horizontal_distance_m=sum(m.horizontal_distance_m for m in steps),
            time_s=sum(m.time_s for m in steps),
            energy_j=sum(m.energy_j for m in steps),
            ascent_m=sum(max(0.0, m.elevation_change_m) for m in steps),
            descent_m=sum(max(0.0, -m.elevation_change_m) for m in steps),
            expanded_nodes=expanded,
        )

    def _search(self, start: Cell, goal: Cell, objective: str) -> OptimizedRoute:
        if objective not in ("time", "energy"):
            raise ValueError("objective must be 'time' or 'energy'")
        start = self._cell(start)
        goal = self._cell(goal)

        # Validate cells and clearance using the actual model's decisions.
        # Out-of-range cells raise IndexError with context from traversability.
        for label, cell in (("Start", start), ("Goal", goal)):
            status = self.model.evaluate_tile(*cell)
            if not status.passable:
                return self._not_found(objective, start, goal,
                                       f"{label} cell {cell} blocked: {status.reason}")

        if start == goal:
            return self._assemble(objective, start, goal, {}, 0)

        # Priority is g+h (A*). Secondary keys make tie-breaking deterministic.
        # States contain position only: battery is intentionally not part of
        # either of these two objectives.
        counter = 0
        fringe = [(self._heuristic(start, goal, objective), counter, 0.0, start)]
        best_g: dict[Cell, float] = {start: 0.0}
        previous: dict[Cell, tuple[Cell, MoveResult]] = {}
        expanded = 0

        while fringe:
            _, _, g, current = heappop(fringe)
            if g > best_g.get(current, inf):
                continue  # outdated entry from an earlier worse route
            if current == goal:
                return self._assemble(objective, start, goal, previous, expanded)

            if self.max_expanded_nodes is not None and expanded >= self.max_expanded_nodes:
                return self._not_found(
                    objective, start, goal,
                    f"Stopped at max_expanded_nodes={self.max_expanded_nodes}; "
                    "a route may still exist", expanded,
                )
            expanded += 1

            for neighbor in self._neighbors(current):
                move = self._move(current, neighbor)
                if not move.passable:
                    continue
                edge_cost = move.time_s if objective == "time" else move.energy_j
                if not isfinite(edge_cost) or edge_cost < 0:
                    raise ValueError("Passable move has invalid (negative/nonfinite) cost")
                proposed = g + edge_cost
                if proposed < best_g.get(neighbor, inf):
                    best_g[neighbor] = proposed
                    previous[neighbor] = (current, move)
                    counter += 1
                    heappush(
                        fringe,
                        (proposed + self._heuristic(neighbor, goal, objective),
                         counter, proposed, neighbor),
                    )

        return self._not_found(objective, start, goal,
                               "No traversable route connects the selected cells", expanded)

    def find_fastest_path(self, start: Cell, goal: Cell) -> OptimizedRoute:
        """Minimize summed `MoveResult.time_s`, with NO charging/stops."""
        return self._search(start, goal, "time")

    def find_lowest_energy_path(self, start: Cell, goal: Cell) -> OptimizedRoute:
        """Minimize summed `MoveResult.energy_j`, with NO charging/stops."""
        return self._search(start, goal, "energy")

    def optimize(self, start: Cell, goal: Cell) -> OptimizationResults:
        """Return both independently optimal routes and full cost summaries."""
        return OptimizationResults(
            fastest=self.find_fastest_path(start, goal),
            lowest_energy=self.find_lowest_energy_path(start, goal),
        )

    # TODO (future project phase): create a separate find_charging_route()
    # whose state is (row, col, charge_bin). That search must track battery
    # capacity, starting charge, each move's energy cost, charger locations,
    # recharging duration/rate, and possibly a maximum time/charge budget.
    # Do NOT simply subtract a recharge time from either route above.


def optimize_paths(
    terrain_or_model,
    start: Cell,
    goal: Cell,
    *,
    config: RoverConfig | None = None,
    material_grid=None,
    materials: Mapping[str, MaterialProperties] | None = None,
    allow_diagonal: bool = True,
    max_expanded_nodes: int | None = None,
) -> OptimizationResults:
    """Convenience entry point for main.py or the UI.

    Pass either an existing TraversabilityModel (recommended if terrain is
    classified already) or a terrain.Terrain object. The latter automatically
    creates TraversabilityModel using the optional surface/config inputs.
    """
    if isinstance(terrain_or_model, TraversabilityModel):
        if config is not None or material_grid is not None or materials is not None:
            raise ValueError("Supply config/materials when constructing the model, not twice")
        model = terrain_or_model
    else:
        model = TraversabilityModel(
            terrain_or_model, config=config,
            material_grid=material_grid, materials=materials,
        )
    return PathOptimizer(
        model, allow_diagonal=allow_diagonal,
        max_expanded_nodes=max_expanded_nodes,
    ).optimize(start, goal)


def route_lonlat(terrain, route: OptimizedRoute) -> list[tuple[float, float]]:
    """Return (lon, lat) cell-centre coordinates for plotting a found route.

    Requires terrain.Terrain.cell_to_lonlat(). A no-route result gives [].
    Coordinates are for the centres of selected 2 m tiles, rather than the
    original exact waypoints. The exact endpoints can be plotted separately.
    """
    return [tuple(map(float, terrain.cell_to_lonlat(row, col)))
            for row, col in route.cells]


def optimize_case(
    case: Mapping,
    *,
    dtm_path: str | None = None,
    margin_m: float = 100.0,
    cell_size: float = 2.0,
    config: RoverConfig | None = None,
    material_grid=None,
    materials: Mapping[str, MaterialProperties] | None = None,
    allow_diagonal: bool = True,
    max_expanded_nodes: int | None = None,
):
    """Load one `data/cases.json` case and optimize both paths.

    Returns (terrain, OptimizationResults). Loads the DTM through the existing
    terrain.load_case() function, so real data availability matters. This
    function is imported only when called; synthetic tests need no DTM.

    The start and end are snapped to the containing 2 m cell by
    terrain.lonlat_to_cell(). The result's distance/time/energy cover the
    cell-centre-to-cell-centre route, not a sub-cell correction at endpoints.
    """
    from terrain import load_case

    terrain = load_case(case, dtm_path=dtm_path,
                        margin_m=margin_m, cell_size=cell_size)
    start = terrain.lonlat_to_cell(case["start"]["lon"], case["start"]["lat"])
    goal = terrain.lonlat_to_cell(case["end"]["lon"], case["end"]["lat"])
    routes = optimize_paths(
        terrain, start, goal,
        config=config, material_grid=material_grid, materials=materials,
        allow_diagonal=allow_diagonal, max_expanded_nodes=max_expanded_nodes,
    )
    return terrain, routes


if __name__ == "__main__":
    # Demonstration: synthetic data, NOT real Jezero terrain or rover telemetry.
    from types import SimpleNamespace
    import numpy as np

    grid = np.zeros((7, 11), dtype=float)
    slopes = np.zeros_like(grid)
    slopes[1:6, 5] = 18.0  # impassable barrier; detour through an open end
    terrain_demo = SimpleNamespace(elev=grid, slope_deg=slopes, cell_size=2.0)
    demo_model = TraversabilityModel(terrain_demo)
    routes = PathOptimizer(demo_model).optimize((3, 1), (3, 9))
    for title, route in (("Fastest", routes.fastest),
                         ("Lowest energy", routes.lowest_energy)):
        print(f"\n{title}: {route.reason}")
        if route.found:
            print("  Cells:", list(route.cells))
            print(f"  Distance: {route.distance_m:.2f} m")
            print(f"  Time: {route.time_s:.1f} s ({route.time_s/60:.2f} min)")
            print(f"  Energy: {route.energy_j:.0f} J ({route.energy_wh:.3f} Wh)")
