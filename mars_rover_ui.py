"""Mars rover route viewer (desktop UI).

This is a demonstration only. Rover locations, terrain, and routes are placeholders;
no real Perseverance data or pathfinding algorithms are included.

Run directly:
    python mars_rover_ui.py

Or from main.py:
    from mars_rover_ui import launch_ui
    launch_ui()

To integrate real data, pass RoverPoint objects, a path_provider function,
and (optionally) an image path and map bounds to launch_ui().
"""

from __future__ import annotations

from dataclasses import dataclass
from math import cos, hypot, radians, sin
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable

try:
    from PIL import Image, ImageTk
except ImportError:  # Pillow is only needed to load an external image.
    Image = None
    ImageTk = None


MIN_DISTANCE_M = 50.0
MAX_DISTANCE_M = 500.0
ROUTE_NAMES = ("Actual rover path", "Fastest path", "Most energy-efficient path", "Balanced path")
ROUTE_STYLES = {
    "Actual rover path": ("#f97316", ()),
    "Fastest path": ("#0284c7", (9, 5)),
    "Most energy-efficient path": ("#9333ea", (5, 4)),
    "Balanced path": ("#16a34a", (2, 5)),
}


@dataclass(frozen=True)
class RoverPoint:
    """A rover position in a shared local coordinate system, measured in meters."""

    name: str
    x_m: float
    y_m: float


PathProvider = Callable[[RoverPoint, RoverPoint], dict[str, list[tuple[float, float]]]]


def example_points() -> list[RoverPoint]:
    """Create *fictional* positions spaced approximately 50 m along a demo track."""
    locations = [RoverPoint("P01", 0.0, 0.0)]
    x, y = 0.0, 0.0
    headings_deg = (20, 55, 10, -25, -40, 10, 50, 60, 5, -35, -20, 15, 45)
    for number, heading in enumerate(headings_deg, start=2):
        x += 50 * cos(radians(heading))
        y += 50 * sin(radians(heading))
        locations.append(RoverPoint(f"P{number:02d}", x, y))
    return locations


def demo_path_provider_factory(points: list[RoverPoint]) -> PathProvider:
    """Return fake routes for testing the UI. These are NOT optimized routes."""
    index_by_name = {point.name: i for i, point in enumerate(points)}

    def make_routes(start: RoverPoint, end: RoverPoint) -> dict[str, list[tuple[float, float]]]:
        first = index_by_name[start.name]
        last = index_by_name[end.name]
        visited = points[first:last + 1] if last >= first else list(reversed(points[last:first + 1]))
        actual = [(p.x_m, p.y_m) for p in visited]

        dx = end.x_m - start.x_m
        dy = end.y_m - start.y_m
        distance = hypot(dx, dy)
        nx, ny = (-dy / distance, dx / distance) if distance else (0.0, 0.0)
        midpoint = ((start.x_m + end.x_m) / 2, (start.y_m + end.y_m) / 2)
        energy_midpoint = (midpoint[0] + nx * 0.13 * distance, midpoint[1] + ny * 0.13 * distance)
        balanced_midpoint = (midpoint[0] - nx * 0.06 * distance, midpoint[1] - ny * 0.06 * distance)
        return {
            "Actual rover path": actual,
            "Fastest path": [(start.x_m, start.y_m), (end.x_m, end.y_m)],
            "Most energy-efficient path": [(start.x_m, start.y_m), energy_midpoint, (end.x_m, end.y_m)],
            "Balanced path": [(start.x_m, start.y_m), balanced_midpoint, (end.x_m, end.y_m)],
        }

    return make_routes


class MarsRoverUI(ttk.Frame):
    """Embeddable Tkinter UI. Construct with a parent Tk window from main.py."""

    def __init__(
        self,
        parent: tk.Misc,
        points: list[RoverPoint] | None = None,
        path_provider: PathProvider | None = None,
        map_image_path: str | Path | None = None,
        map_bounds_m: tuple[float, float, float, float] | None = None,
    ) -> None:
        super().__init__(parent, padding=14)
        self.points = list(points) if points is not None else example_points()
        if len(self.points) < 2 or len({p.name for p in self.points}) != len(self.points):
            raise ValueError("Provide at least two positions with unique names.")
        self.by_name = {p.name: p for p in self.points}
        self.path_provider = path_provider or demo_path_provider_factory(self.points)
        self.demo_mode = path_provider is None
        self.map_bounds_m = map_bounds_m or self._bounds_from_points()
        x0, x1, y0, y1 = self.map_bounds_m
        if not (x0 < x1 and y0 < y1):
            raise ValueError("map_bounds_m must be (min_x, max_x, min_y, max_y).")
        self._loaded_image = None
        self._canvas_image = None
        self._click_positions: dict[str, tuple[float, float]] = {}
        self.routes: dict[str, list[tuple[float, float]]] = {}

        self.start_name = tk.StringVar(value="")
        self.end_name = tk.StringVar(value="")
        self.distance_text = tk.StringVar(value="Distance: select two points")
        self.status_text = tk.StringVar(value="Click two points on the map, or use the menus.")
        self.show_route = {name: tk.BooleanVar(value=True) for name in ROUTE_NAMES}

        self._build_ui()
        if map_image_path is not None:
            self.set_map_image(map_image_path)
        self.after_idle(self._redraw)

    def _bounds_from_points(self) -> tuple[float, float, float, float]:
        x_values = [p.x_m for p in self.points]
        y_values = [p.y_m for p in self.points]
        # Generous margins allow the illustrative alternative routes to remain visible.
        margin = 75.0
        return (min(x_values) - margin, max(x_values) + margin,
                min(y_values) - margin, max(y_values) + margin)

    def _build_ui(self) -> None:
        self.pack(fill="both", expand=True)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(2, weight=1)

        ttk.Label(self, text="MARS ROVER  |  PATH COMPARISON", font=("Segoe UI", 19, "bold")).grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 3)
        )
        ttk.Label(self, text="Prototype: fictional rover points and routes (not real Perseverance data).",
                  foreground="#9a3412").grid(row=1, column=0, columnspan=2, sticky="w", pady=(0, 12))

        map_frame = ttk.Frame(self)
        map_frame.grid(row=2, column=0, sticky="nsew", padx=(0, 16))
        map_frame.columnconfigure(0, weight=1)
        map_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(map_frame, background="#eee2cf", highlightthickness=1,
                                highlightbackground="#c3b7a6", width=800, height=600)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda event: self._redraw())
        self.canvas.bind("<Button-1>", self._map_click)
        ttk.Label(map_frame, text="Map coordinates are local meters; distance is measured in a straight line.",
                  foreground="#475569").grid(row=1, column=0, sticky="w", pady=(8, 0))

        sidebar = ttk.Frame(self, width=290)
        sidebar.grid(row=2, column=1, sticky="ns")
        sidebar.grid_propagate(False)

        ttk.Label(sidebar, text="Choose rover positions", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        ttk.Label(sidebar, text="Start point").pack(anchor="w")
        self.start_combo = ttk.Combobox(sidebar, textvariable=self.start_name, values=list(self.by_name),
                                        state="readonly", width=25)
        self.start_combo.pack(fill="x", pady=(3, 10))
        self.start_combo.bind("<<ComboboxSelected>>", self._selection_changed)

        ttk.Label(sidebar, text="End point").pack(anchor="w")
        self.end_combo = ttk.Combobox(sidebar, textvariable=self.end_name, values=list(self.by_name),
                                      state="readonly", width=25)
        self.end_combo.pack(fill="x", pady=(3, 10))
        self.end_combo.bind("<<ComboboxSelected>>", self._selection_changed)
        ttk.Label(sidebar, textvariable=self.distance_text, wraplength=275,
                  font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(0, 4))
        ttk.Label(sidebar, text="Allowed straight-line distance: 50–500 m (inclusive).",
                  wraplength=270).pack(anchor="w", pady=(0, 12))

        self.calculate_button = ttk.Button(sidebar, text="Calculate Paths", command=self.calculate_paths)
        self.calculate_button.pack(fill="x", pady=(0, 8))
        ttk.Button(sidebar, text="Clear Selection", command=self.clear_selection).pack(fill="x", pady=(0, 18))

        ttk.Separator(sidebar).pack(fill="x", pady=(0, 12))
        ttk.Label(sidebar, text="Show paths", font=("Segoe UI", 12, "bold")).pack(anchor="w", pady=(0, 8))
        for name in ROUTE_NAMES:
            ttk.Checkbutton(sidebar, text=name, variable=self.show_route[name],
                            command=self._redraw).pack(anchor="w", pady=(0, 8))
        ttk.Label(sidebar, text="Orange: recorded/demo track\nBlue: fastest\nPurple: energy efficient\nGreen: balanced",
                  foreground="#475569", justify="left").pack(anchor="w", pady=(4, 10))
        ttk.Label(sidebar, text="Path lines are illustrative until your group supplies real algorithms.",
                  wraplength=270, foreground="#9a3412").pack(anchor="w")

        ttk.Label(self, textvariable=self.status_text, anchor="w", relief="groove", padding=8,
                  wraplength=950).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(12, 0))

    def set_map_image(self, image_path: str | Path) -> None:
        """Replace the background with an image; requires Pillow (`pip install pillow`).

        The image is stretched across map_bounds_m. Supply matching bounds for
        correct alignment with real rover positions.
        """
        if Image is None:
            raise RuntimeError("Install Pillow to use map images: python -m pip install pillow")
        with Image.open(image_path) as source:
            self._loaded_image = source.convert("RGB")
        self._redraw()

    def _world_to_pixel(self, x_m: float, y_m: float) -> tuple[float, float]:
        width = max(self.canvas.winfo_width(), 2)
        height = max(self.canvas.winfo_height(), 2)
        min_x, max_x, min_y, max_y = self.map_bounds_m
        pad = 24
        px = pad + (x_m - min_x) / (max_x - min_x) * max(width - pad * 2, 1)
        py = height - pad - (y_m - min_y) / (max_y - min_y) * max(height - pad * 2, 1)
        return px, py

    def _redraw(self) -> None:
        if not hasattr(self, "canvas"):
            return
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 2)
        height = max(canvas.winfo_height(), 2)
        if width < 70 or height < 70:
            return

        if self._loaded_image is not None and ImageTk is not None:
            self._canvas_image = ImageTk.PhotoImage(self._loaded_image.resize((width, height)))
            canvas.create_image(0, 0, image=self._canvas_image, anchor="nw")
        else:
            # Abstract terrain placeholder. Not an image of the surface of Mars.
            canvas.create_rectangle(0, 0, width, height, fill="#eee2cf", outline="")
            for a, b, radius in [(0.20, 0.25, 55), (0.72, 0.36, 86), (0.40, 0.75, 65), (0.85, 0.80, 38)]:
                cx, cy = a * width, b * height
                canvas.create_oval(cx-radius, cy-radius*0.55, cx+radius, cy+radius*0.55,
                                   outline="#cfbda5", width=3)
            canvas.create_text(18, 16, anchor="nw", text="MAP PLACEHOLDER — no Mars image loaded",
                               fill="#805e42", font=("Segoe UI", 11, "bold"))

        # Display a simple coordinate grid. Axis bounds are local meters.
        min_x, max_x, min_y, max_y = self.map_bounds_m
        for i in range(1, 5):
            x = min_x + i * (max_x - min_x) / 5
            y = min_y + i * (max_y - min_y) / 5
            xpx, _ = self._world_to_pixel(x, min_y)
            _, ypx = self._world_to_pixel(min_x, y)
            canvas.create_line(xpx, 24, xpx, height-24, fill="#bdad9a", dash=(3, 8))
            canvas.create_line(24, ypx, width-24, ypx, fill="#bdad9a", dash=(3, 8))

        for name in ROUTE_NAMES:
            if name not in self.routes or not self.show_route[name].get():
                continue
            route = self.routes[name]
            if len(route) < 2:
                continue
            pixels = [coordinate for xy in route for coordinate in self._world_to_pixel(*xy)]
            color, dash = ROUTE_STYLES[name]
            canvas.create_line(*pixels, fill=color, width=4, dash=dash,
                               capstyle=tk.ROUND, joinstyle=tk.ROUND)

        self._click_positions = {}
        for point in self.points:
            x, y = self._world_to_pixel(point.x_m, point.y_m)
            self._click_positions[point.name] = (x, y)
            selected = point.name == self.start_name.get() or point.name == self.end_name.get()
            color = "#dc2626" if selected else "#1e293b"
            radius = 7 if selected else 5
            canvas.create_oval(x-radius, y-radius, x+radius, y+radius,
                               fill=color, outline="white", width=2)
            canvas.create_text(x+9, y-11, anchor="w", text=point.name,
                               fill="#0f172a", font=("Segoe UI", 9, "bold"))
        canvas.create_text(18, height-12, anchor="sw", fill="#475569",
                           text="Click a point to select START, then click another for END")

    def _map_click(self, event: tk.Event) -> None:
        name = None
        nearest = 16.0
        for key, (x, y) in self._click_positions.items():
            delta = hypot(event.x-x, event.y-y)
            if delta < nearest:
                nearest = delta
                name = key
        if name is None:
            return
        if not self.start_name.get() or self.end_name.get():
            self.start_name.set(name)
            self.end_name.set("")
        else:
            self.end_name.set(name)
        self._selection_changed()

    def _selection_changed(self, _event: object = None) -> None:
        self.routes.clear()  # Old routes must not appear for a new selection.
        start = self.by_name.get(self.start_name.get())
        end = self.by_name.get(self.end_name.get())
        if start is None or end is None:
            self.distance_text.set("Distance: select two points")
            self.status_text.set("Select both a start and an end position.")
        else:
            distance = hypot(end.x_m - start.x_m, end.y_m - start.y_m)
            self.distance_text.set(f"Straight-line distance: {distance:.1f} m")
            valid = start.name != end.name and MIN_DISTANCE_M - 1e-7 <= distance <= MAX_DISTANCE_M + 1e-7
            self.status_text.set("Selection valid. Click Calculate Paths." if valid else
                                 "Choose two different points between 50 m and 500 m apart.")
        self._redraw()

    def calculate_paths(self) -> None:
        start = self.by_name.get(self.start_name.get())
        end = self.by_name.get(self.end_name.get())
        if start is None or end is None:
            messagebox.showwarning("Select points", "Select a start point and an end point first.")
            return
        distance = hypot(end.x_m-start.x_m, end.y_m-start.y_m)
        if start.name == end.name or not (MIN_DISTANCE_M - 1e-7 <= distance <= MAX_DISTANCE_M + 1e-7):
            messagebox.showwarning("Distance outside range",
                                   "Choose different rover positions 50–500 meters apart (straight-line distance).")
            return
        try:
            routes = self.path_provider(start, end)
            if not isinstance(routes, dict):
                raise TypeError("The path provider must return a dictionary of named paths.")
            checked: dict[str, list[tuple[float, float]]] = {}
            for name, path in routes.items():
                if name not in ROUTE_NAMES:
                    continue
                coords = [(float(x), float(y)) for x, y in path]
                if len(coords) >= 2:
                    checked[name] = coords
            if not checked:
                raise ValueError("No valid paths were returned by the path provider.")
        except Exception as error:
            messagebox.showerror("Path calculation failed", str(error))
            return
        self.routes = checked
        qualifier = "DEMO ONLY: illustrative paths" if self.demo_mode else "Routes returned by path provider"
        self.status_text.set(f"{qualifier} for {start.name} → {end.name} ({distance:.1f} m). "
                             "Use the checkboxes to show or hide them.")
        self._redraw()

    def clear_selection(self) -> None:
        self.start_name.set("")
        self.end_name.set("")
        self.routes.clear()
        self.distance_text.set("Distance: select two points")
        self.status_text.set("Click two points on the map, or use the menus.")
        self._redraw()


def launch_ui(
    points: list[RoverPoint] | None = None,
    path_provider: PathProvider | None = None,
    map_image_path: str | Path | None = None,
    map_bounds_m: tuple[float, float, float, float] | None = None,
) -> None:
    """Open this module as its own desktop window, including when called by main.py."""
    root = tk.Tk()
    root.title("Mars Rover — Path Comparison")
    root.geometry("1210x760")
    root.minsize(870, 570)
    MarsRoverUI(root, points=points, path_provider=path_provider,
                map_image_path=map_image_path, map_bounds_m=map_bounds_m)
    root.mainloop()


if __name__ == "__main__":
    launch_ui()
