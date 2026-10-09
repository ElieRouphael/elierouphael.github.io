"""Find a lost robot with a particle filter.

A robot drives over a hilly map. It has no GPS. It has a compass, so it knows
which way it is facing, and an altimeter, which measures the height of the
ground under it. We spread 2,000 guesses ("particles") over the map and, at
every step, move them, weigh them and resample them until they agree on where
the robot is.

Companion code for the tutorial at
https://elierouphael.github.io/tutorials/particle-filter/

Needs Python 3 with numpy and matplotlib:   pip install numpy matplotlib
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(seed=1)    # change the seed for a new map and a new drive

# ------------------------------------------------------------------ the world
WIDTH, HEIGHT = 240, 160               # map size in metres (one grid cell = 1 m)


def make_terrain(width, height):
    """Random hills: smooth noise at five scales, added together."""
    ys, xs = np.mgrid[0:height, 0:width]
    terrain = np.zeros((height, width))
    for octave in range(5):
        cells = 3 * 2**octave                        # 3, 6, 12, 24, 48 bumps across
        grid = rng.uniform(-1, 1, (cells + 2, cells + 2))
        gx, gy = xs * cells / width, ys * cells / width
        ix, iy = gx.astype(int), gy.astype(int)
        fx, fy = gx - ix, gy - iy
        fx, fy = fx * fx * (3 - 2 * fx), fy * fy * (3 - 2 * fy)   # smooth blend
        top = grid[iy, ix] * (1 - fx) + grid[iy, ix + 1] * fx
        bottom = grid[iy + 1, ix] * (1 - fx) + grid[iy + 1, ix + 1] * fx
        terrain += 0.55**octave * (top * (1 - fy) + bottom * fy)
    terrain -= terrain.min()
    return 100 * terrain / terrain.max()             # heights from 0 to 100 m


terrain = make_terrain(WIDTH, HEIGHT)


def height_at(x, y):
    """Ground height at (x, y). Works on one position or on arrays of them."""
    col = np.clip(np.rint(x).astype(int), 0, WIDTH - 1)
    row = np.clip(np.rint(y).astype(int), 0, HEIGHT - 1)
    return terrain[row, col]                         # images are indexed [row, column]


# ------------------------------------------------------------------ the robot
STEP_NOISE = 0.10      # the wheels get each distance wrong by about 10 %
TURN_NOISE = 0.10      # ... and each turn by about 10 %
DRIFT = 0.015          # driving also wobbles the heading (radians per metre)
COMPASS_NOISE = 0.03   # the compass is off by about 2 degrees
SENSOR_NOISE = 1.5     # the altimeter is off by about 1.5 m


def robot_step(x, y, heading, distance, turn):
    """The real robot turns, then drives forward. Nothing is ever exact."""
    heading = heading + turn + rng.normal(0, TURN_NOISE * abs(turn) + DRIFT * distance)
    travelled = distance + rng.normal(0, STEP_NOISE * distance)
    x = np.clip(x + travelled * np.cos(heading), 0, WIDTH - 1)    # a fence keeps the
    y = np.clip(y + travelled * np.sin(heading), 0, HEIGHT - 1)   # robot on the map
    return x, y, heading


def read_compass(heading):
    return heading + rng.normal(0, COMPASS_NOISE)


def read_altimeter(x, y):
    return height_at(x, y) + rng.normal(0, SENSOR_NOISE)


# --------------------------------------------------------- the particle filter
N = 2000                 # number of guesses
ASSUMED_NOISE = 2.5      # the filter assumes a slightly worse altimeter than the real one
JITTER = 0.5             # metres of shake after resampling


def predict(px, py, compass, distance):
    """1. Move every guess the way the robot thinks it moved, plus random errors."""
    heading = compass + rng.normal(0, COMPASS_NOISE + DRIFT * distance, N)
    travelled = distance + rng.normal(0, STEP_NOISE * distance, N)
    px = np.clip(px + travelled * np.cos(heading), 0, WIDTH - 1)
    py = np.clip(py + travelled * np.sin(heading), 0, HEIGHT - 1)
    return px, py


def weigh(px, py, reading):
    """2. Bell-curve score: how well does each guess explain the reading?"""
    surprise = ((reading - height_at(px, py)) / ASSUMED_NOISE) ** 2
    weights = np.exp(-0.5 * (surprise - surprise.min()))   # best guess gets weight 1
    return weights / weights.sum()                          # weights now add up to 1


def resample(px, py, weights):
    """3. Copy guesses in proportion to their weight; unlikely ones disappear."""
    comb = (rng.random() + np.arange(N)) / N               # N evenly spaced teeth
    chosen = np.searchsorted(np.cumsum(weights), comb)
    chosen = np.minimum(chosen, N - 1)                      # guard against rounding
    return px[chosen], py[chosen]


def jitter(px, py):
    """4. A small shake, so that copies do not sit exactly on top of each other."""
    return px + rng.normal(0, JITTER, N), py + rng.normal(0, JITTER, N)


# ------------------------------------------------------------------ the drive
# The robot starts somewhere on the map; the guesses start everywhere.
rx, ry, rh = 40.0, 80.0, 0.0
px = rng.uniform(0, WIDTH, N)
py = rng.uniform(0, HEIGHT, N)

STEPS = 40
snapshots = {0: (px, py, rx, ry)}
errors, spreads = [], []

for step in range(1, STEPS + 1):
    distance, turn = 4.0, 0.2 * np.cos(step / 4)           # weave left and right

    rx, ry, rh = robot_step(rx, ry, rh, distance, turn)    # the real robot moves
    compass = read_compass(rh)                              # and reads its sensors
    reading = read_altimeter(rx, ry)

    px, py = predict(px, py, compass, distance)             # 1. predict
    weights = weigh(px, py, reading)                        # 2. weigh
    px, py = resample(px, py, weights)                      # 3. resample
    px, py = jitter(px, py)                                 # 4. jitter

    guess_x, guess_y = px.mean(), py.mean()                 # the best single guess
    errors.append(np.hypot(guess_x - rx, guess_y - ry))
    spreads.append(np.sqrt(px.var() + py.var()))
    if step in (1, 4, STEPS):
        snapshots[step] = (px, py, rx, ry)
    print(f"step {step:2d}   altimeter {reading:5.1f} m   "
          f"error {errors[-1]:6.1f} m   spread {spreads[-1]:6.1f} m")

# ------------------------------------------------------------------ the plots
fig, axes = plt.subplots(1, len(snapshots), figsize=(16, 3.6))
for ax, (step, (sx, sy, x, y)) in zip(axes, snapshots.items()):
    ax.imshow(terrain, cmap="gray", origin="upper")        # dark = low, light = high
    ax.scatter(sx, sy, s=1, color="#2a78d6", label="guesses")
    ax.scatter([x], [y], s=60, color="#eb6834", edgecolor="black", label="real robot")
    ax.set_xlim(0, WIDTH - 1)
    ax.set_ylim(HEIGHT - 1, 0)
    ax.set_title(f"after {step} steps")
    ax.set_xticks([])
    ax.set_yticks([])
axes[0].legend(loc="lower left", fontsize=8)
fig.tight_layout()

fig2, ax2 = plt.subplots(figsize=(7, 3))
ax2.plot(range(1, STEPS + 1), errors, color="#eb6834", label="error of the best guess")
ax2.plot(range(1, STEPS + 1), spreads, color="#2a78d6", label="spread of the guesses")
ax2.set_xlabel("step")
ax2.set_ylabel("metres (log scale)")
ax2.set_yscale("log")
ax2.legend()
fig2.tight_layout()
plt.show()
