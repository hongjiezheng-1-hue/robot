"""The unknown obstacle course used for the navigation experiments and the demonstration."""

# three thin boards, staggered so that the route weaves up, down and up again (x, y, size_x, size_y) [m]
S_COURSE = ((-1.5, 0.65, 0.10, 1.70), (-0.5, -0.65, 0.10, 1.70), (0.5, 0.65, 0.10, 1.70))
START = (-1.65, -0.6, 0.0)       # nominal start pose the robot believes it starts at (x, y, heading)
