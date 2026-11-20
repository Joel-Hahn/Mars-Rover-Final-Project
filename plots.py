"""plots.py: how each route uses energy and covers distance over time.

Owner: Shai. Draws two graphs for the NASA recorded path, the fastest path
and the lowest-energy path:
  * energy used (Wh) against elapsed time (minutes)
  * distance travelled (m) against elapsed time (minutes)

Every route is a list of 2 m steps ("moves") from traversability.py. Each
step has its own time_s, energy_j and distance_m, so the graphs are just
running totals of those steps.

Use from the UI:
    import plots
    profiles = [plots.route_profile(name, moves, note) for ...]
    plots.open_plot_window(root, profiles, ROUTE_STYLES)

Or on its own, to save a PNG for a report:
    python plots.py --case case1 --out route_graphs.png

All times and energies are MODEL ESTIMATES from traversability.py, not NASA
telemetry.
"""

from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

# Fallback colours when no styles are given (fixed order, never cycled).
_DEFAULT_COLORS = ("#2a78d6", "#eb6834", "#1baf7a")
_INK, _MUTED, _GRID = "#0b0b0b", "#52514e", "#e1e0d9"


@dataclass(frozen=True)
class RouteProfile:
    """Running totals along one route. Index 0 is the start (all zeros)."""

    name: str
    time_s: np.ndarray        # elapsed time after each step (s)
    energy_wh: np.ndarray     # energy used so far (Wh)
    distance_m: np.ndarray    # distance travelled so far (m)
    note: str = ""            # e.g. why a route is unavailable

    @property
    def available(self) -> bool:
        """False when the route has no steps (not found, or not modelled)."""
        return len(self.time_s) > 1

    @property
    def total_time_s(self) -> float:
        return float(self.time_s[-1])

    @property
    def total_energy_wh(self) -> float:
        return float(self.energy_wh[-1])

    @property
    def total_distance_m(self) -> float:
        return float(self.distance_m[-1])


def route_profile(name, moves, note=""):
    """Turn a route's steps into running totals of time, energy and distance.

    moves : steps with .time_s, .energy_j and .distance_m, e.g.
            OptimizedRoute.moves or PathResult.moves. May be empty.
    """
    moves = list(moves)
    steps = np.array([[m.time_s, m.energy_j, m.distance_m] for m in moves],
                     dtype=float).reshape(-1, 3)
    if not np.all(np.isfinite(steps)):
        raise ValueError(f"{name}: step time/energy/distance must be finite")
    if np.any(steps < 0):
        raise ValueError(f"{name}: step time/energy/distance can't be negative")

    # Running totals, with a 0 in front so every line starts at the origin.
    totals = np.vstack([np.zeros((1, 3)), np.cumsum(steps, axis=0)])
    return RouteProfile(name=name, time_s=totals[:, 0],
                        energy_wh=totals[:, 1] / 3600.0,      # J -> Wh
                        distance_m=totals[:, 2], note=note)


def make_figure(profiles, styles=None):
    """Two side-by-side graphs, one line per available route.

    styles : optional {route name: (colour, dash pattern)}, e.g. the UI's
             ROUTE_STYLES, so each route has the same look as on the map.
    Uses matplotlib's Figure directly (not pyplot), so it can be embedded
    in Tk and drawn from any thread without global state.
    """
    styles = styles or {}
    fig = Figure(figsize=(10, 4.4), layout="constrained")
    ax_energy, ax_dist = fig.subplots(1, 2)

    shown = [p for p in profiles if p.available]
    missing = [p for p in profiles if not p.available]

    for i, p in enumerate(shown):
        color, dashes = styles.get(p.name, (_DEFAULT_COLORS[i % 3], ()))
        minutes = p.time_s / 60.0
        for ax, values in ((ax_energy, p.energy_wh), (ax_dist, p.distance_m)):
            line, = ax.plot(minutes, values, color=color, lw=2, label=p.name)
            if dashes:
                line.set_dashes(dashes)

    # Direct label at the end of each line: the route's total.
    for ax, unit, totals in ((ax_energy, "Wh", [p.total_energy_wh for p in shown]),
                             (ax_dist, "m", [p.total_distance_m for p in shown])):
        _label_line_ends(ax, [p.time_s[-1] / 60.0 for p in shown], totals, unit)

    for ax, title, ylabel in ((ax_energy, "Energy used over time", "Energy used (Wh)"),
                              (ax_dist, "Distance travelled over time",
                               "Distance travelled (m)")):
        ax.set_title(title, loc="left", color=_INK, fontsize=11)
        ax.set_xlabel("Time (min)", color=_MUTED)
        ax.set_ylabel(ylabel, color=_MUTED)
        ax.tick_params(colors=_MUTED, labelsize=9)
        ax.grid(color=_GRID, lw=0.8)
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.margins(x=0.12)                       # room for the end labels
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(_GRID)

    if shown:
        ax_energy.legend(loc="upper left", fontsize=9, frameon=False)
    else:
        fig.text(0.5, 0.5, "No routes to show yet: calculate routes first.",
                 ha="center", color=_MUTED)
    if missing:
        lines = [f"{p.name}: not shown. {p.note}".strip() for p in missing]
        fig.text(0.01, 0.005, "\n".join(lines), fontsize=8, color=_MUTED,
                 va="bottom")
        fig.get_layout_engine().set(rect=(0, 0.04 * len(lines), 1, 1))
    fig.suptitle("Model estimates from traversability.py, not NASA telemetry",
                 x=0.99, ha="right", fontsize=8, color=_MUTED)
    return fig


def _label_line_ends(ax, xs, ys, unit):
    """Write each line's total at its end, nudging labels apart if they'd overlap."""
    ax.autoscale_view()                         # make the axis limits final
    y_lo, y_hi = ax.get_ylim()
    x_lo, x_hi = ax.get_xlim()
    min_gap = 0.05 * (y_hi - y_lo)              # about one line of text
    x_pad = 0.01 * (x_hi - x_lo)                # small space after the line end

    last_label_y = -np.inf
    for i in np.argsort(ys):                    # lowest total first
        label_y = max(ys[i], last_label_y + min_gap)   # push up if too close
        ax.annotate(f"{ys[i]:.1f} {unit}", xy=(xs[i], ys[i]),
                    xytext=(xs[i] + x_pad, label_y), textcoords="data",
                    va="center", fontsize=8, color=_MUTED, annotation_clip=False)
        last_label_y = label_y


def save_figure(profiles, path, styles=None, dpi=150):
    """Save the two graphs as an image file (e.g. PNG for a report)."""
    make_figure(profiles, styles).savefig(path, dpi=dpi)


def open_plot_window(parent, profiles, styles=None, title="Route graphs"):
    """Open the graphs in a new Tk window over `parent`; return the window.

    Includes matplotlib's toolbar (zoom, pan, save). The figure is kept as
    window.figure so it stays alive while the window is open.
    """
    import tkinter as tk
    from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                                   NavigationToolbar2Tk)

    window = tk.Toplevel(parent)
    window.title(title)
    window.figure = make_figure(profiles, styles)
    canvas = FigureCanvasTkAgg(window.figure, master=window)
    toolbar = NavigationToolbar2Tk(canvas, window, pack_toolbar=False)
    toolbar.update()
    toolbar.pack(side="bottom", fill="x")
    canvas.get_tk_widget().pack(side="top", fill="both", expand=True)
    canvas.draw()
    return window


# ---------- command line: graphs for a whole case from cases.json ----------

def profiles_for_case(case, dtm_path=None):
    """Compute all three routes for one case and return their profiles.

    Uses the case's own start/end and its full NASA route (the UI instead
    uses the two markers you pick along that route).
    """
    from mars_rover_ui import (ACTUAL, FASTEST, LOWEST, projected_track,
                               route_cells_near_recorded_line)
    from optimization import PathOptimizer
    from terrain import load_case
    from traversability import TraversabilityModel

    terrain = load_case(case, dtm_path=dtm_path)
    model = TraversabilityModel(terrain)

    # NASA's recorded path, snapped to the 2 m grid and costed by the model.
    track = projected_track(terrain, case["nasa_route_lonlat"])
    nasa = model.evaluate_path(route_cells_near_recorded_line(terrain, track))
    profiles = [route_profile(ACTUAL, nasa.moves if nasa.passable else [],
                              "" if nasa.passable else nasa.reason)]

    start = terrain.lonlat_to_cell(case["start"]["lon"], case["start"]["lat"])
    goal = terrain.lonlat_to_cell(case["end"]["lon"], case["end"]["lat"])
    result = PathOptimizer(model).optimize(start, goal)
    for name, route in ((FASTEST, result.fastest), (LOWEST, result.lowest_energy)):
        profiles.append(route_profile(name, route.moves,
                                      "" if route.found else route.reason))
    return profiles


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Energy and distance over time "
                                                 "for one case's three routes.")
    parser.add_argument("--case", required=True, help="case id, e.g. case1")
    parser.add_argument("--dtm", help="DTM file (default: same rules as terrain.py)")
    parser.add_argument("--out", default="route_graphs.png")
    args = parser.parse_args()

    import cases
    from mars_rover_ui import ROUTE_STYLES

    if not cases.CASES_FILE.exists():
        parser.error(f"{cases.CASES_FILE} not found. Run `py cases.py` first.")
    all_cases = json.loads(cases.CASES_FILE.read_text())
    case = next((c for c in all_cases if c["id"] == args.case), None)
    if case is None:
        parser.error(f"no {args.case!r}; have {[c['id'] for c in all_cases]}")

    profiles = profiles_for_case(case, args.dtm)
    for p in profiles:
        if p.available:
            print(f"{p.name:20} {p.total_time_s / 60:6.1f} min  "
                  f"{p.total_energy_wh:7.2f} Wh  {p.total_distance_m:7.1f} m")
        else:
            print(f"{p.name:20} not available: {p.note}")
    save_figure(profiles, args.out, ROUTE_STYLES)
    print(f"Saved {args.out}")
