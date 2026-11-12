
import tkinter as tk
from tkinter import ttk
import math

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# -----------------------------------
# EXAMPLE MARS ROVER ROUTES
# -----------------------------------

routes = {
    "Shortest Route": [
        (0, 0), (2, 1), (4, 2),
        (6, 2), (8, 4), (10, 5)
    ],

    "Northern Route": [
        (0, 0), (1, 3), (3, 5),
        (6, 6), (8, 6), (10, 5)
    ],

    "Southern Route": [
        (0, 0), (2, -2), (5, -3),
        (8, -1), (10, 5)
    ]
}


# -----------------------------------
# CREATE APPLICATION WINDOW
# -----------------------------------

root = tk.Tk()

root.title("Mars Rover Route Planner")
root.geometry("1100x650")
root.minsize(850, 500)

# Main title
title = ttk.Label(
    root,
    text="Mars Rover Route Planner",
    font=("Arial", 22, "bold")
)
title.pack(pady=15)

# Main container
main_frame = ttk.Frame(root, padding=15)
main_frame.pack(fill="both", expand=True)


# -----------------------------------
# LEFT SIDE: USER CONTROLS
# -----------------------------------

control_frame = ttk.Frame(main_frame, padding=15)
control_frame.pack(side="left", fill="y", padx=10)

ttk.Label(
    control_frame,
    text="Route Settings",
    font=("Arial", 16, "bold")
).pack(pady=10)

# Route selection
ttk.Label(
    control_frame,
    text="Select a Route:"
).pack(anchor="w", pady=5)

selected_route = tk.StringVar(value="Shortest Route")

route_menu = ttk.Combobox(
    control_frame,
    textvariable=selected_route,
    values=list(routes.keys()),
    state="readonly",
    width=22
)
route_menu.pack(pady=5)

# Checkbox
show_all = tk.BooleanVar(value=True)

show_checkbox = ttk.Checkbutton(
    control_frame,
    text="Show All Routes",
    variable=show_all
)
show_checkbox.pack(pady=15)

# Distance output
distance_text = tk.StringVar(
    value="Click Calculate Path"
)

distance_label = ttk.Label(
    control_frame,
    textvariable=distance_text,
    font=("Arial", 12)
)
distance_label.pack(pady=15)


# -----------------------------------
# RIGHT SIDE: GRAPH
# -----------------------------------

graph_frame = ttk.Frame(main_frame)
graph_frame.pack(
    side="right",
    fill="both",
    expand=True,
    padx=10
)

# Create Matplotlib figure
fig, ax = plt.subplots(figsize=(7, 5))

# Embed graph inside Tkinter
canvas = FigureCanvasTkAgg(
    fig,
    master=graph_frame
)
canvas.get_tk_widget().pack(
    fill="both",
    expand=True
)


# -----------------------------------
# UPDATE GRAPH FUNCTION
# -----------------------------------

def update_graph():

    ax.clear()

    current_route = selected_route.get()

    for name, path in routes.items():

        if name != current_route and not show_all.get():
            continue

        x, y = zip(*path)

        if name == current_route:
            ax.plot(
                x, y,
                "-o",
                linewidth=3,
                label=name
            )
        else:
            ax.plot(
                x, y,
                "--",
                color="gray",
                alpha=0.5,
                label=name
            )

    # Starting point
    ax.scatter(
        0, 0,
        color="green",
        s=120,
        label="Start"
    )

    # Destination
    ax.scatter(
        10, 5,
        color="red",
        s=120,
        label="Destination"
    )

    ax.set_title("Mars Rover Route Map")
    ax.set_xlabel("X Position (map units)")
    ax.set_ylabel("Y Position (map units)")

    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    canvas.draw()


# -----------------------------------
# CALCULATE PATH FUNCTION
# -----------------------------------

def calculate_path():

    current_route = selected_route.get()

    points = routes[current_route]

    # Calculate total distance
    distance = sum(
        math.dist(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )

    # Display result
    distance_text.set(
        f"Route Distance: {distance:.2f} units"
    )

    update_graph()


# -----------------------------------
# BUTTON
# -----------------------------------

calculate_button = ttk.Button(
    control_frame,
    text="Calculate Path",
    command=calculate_path
)

calculate_button.pack(pady=10)


# -----------------------------------
# UPDATE WHEN SETTINGS CHANGE
# -----------------------------------

def route_changed(event=None):
    distance_text.set("Click Calculate Path")
    update_graph()


route_menu.bind(
    "<<ComboboxSelected>>",
    route_changed
)

show_checkbox.config(command=update_graph)


# -----------------------------------
# START APPLICATION
# -----------------------------------

update_graph()

root.mainloop()
