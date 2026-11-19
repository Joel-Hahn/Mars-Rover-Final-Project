"""Tests for plots.py: energy-vs-time and distance-vs-time route graphs."""

from collections import namedtuple

import matplotlib

matplotlib.use("Agg")                     # draw to files, never open a window

import numpy as np
import pytest

from plots import make_figure, open_plot_window, route_profile, save_figure

# Anything with time_s, energy_j and distance_m works as a step, like
# traversability.MoveResult does.
Step = namedtuple("Step", "time_s energy_j distance_m")

# 3 steps: 10 s / 1 Wh / 2 m, then 20 s / 2 Wh / 2.83 m (a diagonal),
# then 10 s / 0 Wh / 2 m (downhill, no energy charged).
STEPS = [Step(10.0, 3600.0, 2.0), Step(20.0, 7200.0, 2.83), Step(10.0, 0.0, 2.0)]


# ---------- route_profile ----------

def test_profile_is_running_totals_starting_at_zero():
    p = route_profile("Fastest path", STEPS)
    assert p.time_s.tolist() == [0.0, 10.0, 30.0, 40.0]
    assert p.energy_wh.tolist() == [0.0, 1.0, 3.0, 3.0]
    assert np.allclose(p.distance_m, [0.0, 2.0, 4.83, 6.83])
    assert p.available


def test_profile_totals():
    p = route_profile("Fastest path", STEPS)
    assert p.total_time_s == 40.0
    assert p.total_energy_wh == 3.0
    assert p.total_distance_m == pytest.approx(6.83)


def test_route_with_no_steps_is_unavailable():
    # e.g. no route found, or NASA's path rejected by the cost model
    p = route_profile("NASA recorded path", [], note="Model rejects a tile.")
    assert not p.available
    assert p.note == "Model rejects a tile."


def test_bad_step_values_raise():
    with pytest.raises(ValueError, match="finite"):
        route_profile("Fastest path", [Step(float("inf"), 1.0, 2.0)])
    with pytest.raises(ValueError, match="negative"):
        route_profile("Fastest path", [Step(-1.0, 1.0, 2.0)])


def test_profile_from_real_optimizer_matches_its_totals():
    # A tiny flat 6 x 6 terrain with a 1 m hill in the middle.
    from optimization import PathOptimizer
    from traversability import TraversabilityModel

    class FlatTerrain:
        cell_size = 2.0
        elev = np.zeros((6, 6))
        slope_deg = np.zeros((6, 6))

    FlatTerrain.elev[2:4, 2:4] = 0.5
    result = PathOptimizer(TraversabilityModel(FlatTerrain())).optimize((0, 0), (5, 5))
    for route in (result.fastest, result.lowest_energy):
        assert route.found
        p = route_profile(route.objective, route.moves)
        assert p.total_time_s == pytest.approx(route.time_s)
        assert p.total_energy_wh == pytest.approx(route.energy_wh)
        assert p.total_distance_m == pytest.approx(route.distance_m)
        assert np.all(np.diff(p.time_s) > 0)        # time always moves forward


# ---------- make_figure ----------

STYLES = {"NASA recorded path": ("#ea580c", ()),
          "Fastest path": ("#0284c7", (8, 4)),
          "Lowest-energy path": ("#8b5cf6", (5, 4))}


def _profiles():
    slow = [Step(30.0, 1800.0, 2.0)] * 4           # slower, cheaper
    return [route_profile("Fastest path", STEPS),
            route_profile("Lowest-energy path", slow),
            route_profile("NASA recorded path", [], note="Model rejects a tile.")]


def _lines(ax):
    return {ln.get_label(): ln for ln in ax.get_lines()
            if not ln.get_label().startswith("_")}


def test_figure_has_energy_and_distance_graphs():
    fig = make_figure(_profiles(), STYLES)
    energy_ax, distance_ax = fig.axes
    assert "Energy" in energy_ax.get_ylabel() and "Wh" in energy_ax.get_ylabel()
    assert "Distance" in distance_ax.get_ylabel() and "m" in distance_ax.get_ylabel()
    assert "min" in energy_ax.get_xlabel() and "min" in distance_ax.get_xlabel()


def test_one_line_per_available_route_in_its_ui_colour():
    fig = make_figure(_profiles(), STYLES)
    for ax in fig.axes:
        lines = _lines(ax)
        assert set(lines) == {"Fastest path", "Lowest-energy path"}   # NASA left out
        assert matplotlib.colors.to_hex(lines["Fastest path"].get_color()) == "#0284c7"


def test_lines_end_at_the_route_totals_in_minutes():
    fig = make_figure(_profiles(), STYLES)
    energy_ax, distance_ax = fig.axes
    fast_e = _lines(energy_ax)["Fastest path"]
    assert fast_e.get_xdata()[-1] == pytest.approx(40.0 / 60.0)    # 40 s
    assert fast_e.get_ydata()[-1] == pytest.approx(3.0)            # 3 Wh
    fast_d = _lines(distance_ax)["Fastest path"]
    assert fast_d.get_ydata()[-1] == pytest.approx(6.83)


def test_unavailable_route_is_named_with_its_reason():
    fig = make_figure(_profiles(), STYLES)
    text = " ".join(t.get_text() for t in fig.texts)
    assert "NASA recorded path" in text and "Model rejects a tile." in text


def test_figure_works_without_styles_or_routes():
    assert len(make_figure(_profiles()).axes) == 2           # default colours
    fig = make_figure([])                                     # nothing calculated
    assert "No routes" in " ".join(t.get_text() for t in fig.texts)


def test_save_figure_writes_a_png(tmp_path):
    out = tmp_path / "graphs.png"
    save_figure(_profiles(), out, STYLES)
    assert out.exists() and out.stat().st_size > 10_000


# ---------- open_plot_window (the UI pop-up) ----------

def test_open_plot_window_embeds_the_figure_in_tk():
    tk = pytest.importorskip("tkinter")
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("no display available")
    root.withdraw()
    try:
        win = open_plot_window(root, _profiles(), STYLES, title="Case 1 graphs")
        assert isinstance(win, tk.Toplevel)
        assert win.title() == "Case 1 graphs"
        assert len(win.figure.axes) == 2          # the figure is kept on the window
    finally:
        root.destroy()
