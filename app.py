
import math
import streamlit as st
import matplotlib.pyplot as plt

# Configure the application
st.set_page_config(page_title="Mars Rover", layout="wide")

# Title and introduction
st.title("Mars Rover Route Planner")
st.write("Explore different paths for a Mars rover.")

# Example routes (not real Mars data)
routes = {
    "Shortest Route": [(0, 0), (2, 1), (4, 2), (6, 2), (8, 4), (10, 5)],
    "Northern Route": [(0, 0), (1, 3), (3, 5), (6, 6), (8, 6), (10, 5)],
    "Southern Route": [(0, 0), (2, -2), (5, -3), (8, -1), (10, 5)]
}

# Sidebar controls
st.sidebar.header("Route Settings")

selected_route = st.sidebar.selectbox(
    "Choose a route:",
    list(routes.keys())
)

show_all = st.sidebar.checkbox(
    "Show all routes",
    value=True
)

# Calculate Path button
calculate_button = st.sidebar.button("Calculate Path")

# Store calculation results between interactions
if "route_result" not in st.session_state:
    st.session_state.route_result = None

# Calculate the distance when the button is clicked
if calculate_button:
    points = routes[selected_route]

    distance = sum(
        math.dist(points[i], points[i + 1])
        for i in range(len(points) - 1)
    )

    st.session_state.route_result = {
        "route": selected_route,
        "distance": distance
    }

# Create two columns for the interface
left, right = st.columns([3, 1])

# Left column: Rover path graph
with left:
    st.subheader("Rover Path Map")

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, path in routes.items():
        if name != selected_route and not show_all:
            continue

        x, y = zip(*path)

        if name == selected_route:
            ax.plot(x, y, "-o", linewidth=3, label=name)
        else:
            ax.plot(
                x, y, "--",
                color="gray",
                alpha=0.5,
                label=name
            )

    # Mark the starting point and destination
    ax.scatter(0, 0, color="green", s=120, label="Start")
    ax.scatter(10, 5, color="red", s=120, label="Destination")

    # Graph formatting
    ax.set_xlabel("X Position (map units)")
    ax.set_ylabel("Y Position (map units)")
    ax.set_title("Possible Rover Routes")
    ax.grid(True)
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

# Right column: Route information
with right:
    st.subheader("Route Information")
    st.write("Selected:", selected_route)

    result = st.session_state.route_result

    if result is not None and result["route"] == selected_route:
        st.metric(
            "Route Length",
            f"{result['distance']:.2f} units"
        )
        st.success("Path calculated successfully!")
    else:
        st.info("Click Calculate Path to calculate the route distance.")

    st.info(
        "These are example paths. Real Mars terrain data "
        "will be added later."
    )
