import rospy
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.cm as cm
import sys, os

sys.path.append(os.path.abspath("/mnt/xhand1/dexterous_hands/src/"))
from xhand_utils.src.xhand_utils.xhand_utils import XHandState, XHandStateArrayMsg

# ── Sensor geometry ────────────────────────────────────────────────────────────
N_ROWS, N_COLS = 12, 10
FINGER_NAMES = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

# Helix-corrected physical spacing (from calibration drag data)
COL_SPACING = 4.25   # mm per col  (axial, proximal→distal)
ROW_SPACING = 3.53   # mm per row  (circumferential)
HELIX       = 0.25   # circumferential rows shift per axial col

# Elliptical cross-section semi-axes (from STL analysis)
A_SEMI = 8.5e-3   # m, lateral (X)
B_SEMI = 10.0e-3  # m, front/back (Y, +Y = fingerpad)

def build_taxel_coords():
    """Build physical 2D (axial, circ) and 3D (xyz on cylinder) taxel positions."""
    tx = np.zeros((N_ROWS, N_COLS))  # axial mm (proximal=0, distal=~50)
    ty = np.zeros((N_ROWS, N_COLS))  # circumferential mm
    for r in range(N_ROWS):
        for c in range(N_COLS):
            tx[r, c] = c * COL_SPACING + r * HELIX * COL_SPACING
            ty[r, c] = r * ROW_SPACING - c * HELIX * ROW_SPACING  # minus = correct helix slant

    # 3D: map circumferential mm → angle on ellipse
    # row 0 ≈ -135° (right side), row 11 ≈ +135° (left side), front = +90°
    circ_span = (N_ROWS - 1) * ROW_SPACING
    t3 = np.zeros((N_ROWS, N_COLS, 3))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            angle = np.radians(ty[r, c] / circ_span * 270 - 135)
            t3[r, c, 0] = A_SEMI * np.cos(angle)    # X (lateral)
            t3[r, c, 1] = B_SEMI * np.sin(angle)    # Y (front = +)
            t3[r, c, 2] = tx[r, c] * 1e-3           # Z (proximal=0, distal=+)
    return tx, ty, t3

TAXEL_X, TAXEL_Y, TAXEL_3D = build_taxel_coords()

# ── Layout ─────────────────────────────────────────────────────────────────────
# Top row: 5 × corrected flat unwrap (parallelogram heatmap + quiver)
# Bottom row: 5 × 3D cylinder view
fig = plt.figure(figsize=(20, 10))
gs = gridspec.GridSpec(2, 5, figure=fig, hspace=0.35, wspace=0.25)

flat_axes = []
cyl_axes  = []
heatmaps  = []
quivers   = []
surf_plots = []
quiv3d    = []

# Flat display: 90° CW rotation — circ on x-axis, axial on y-axis
# Proximal at top (y=max), distal at bottom (y=0)
TX_MAX = TAXEL_X.max()
flat_x = TAXEL_Y.flatten()               # circumferential → x
flat_y = (TX_MAX - TAXEL_X).flatten()    # axial flipped → y (proximal=top)

VMAX_FN = 100.0  # tune to your sensor's normal force range

for i in range(5):
    # ── Flat unwrap ────────────────────────────────────────────────────────────
    ax_f = fig.add_subplot(gs[0, i])
    ax_f.set_title(FINGER_NAMES[i], fontsize=10)
    ax_f.set_xlabel('Circ (mm)', fontsize=7)
    ax_f.set_ylabel('Axial: proximal↑ distal↓', fontsize=7)
    ax_f.tick_params(labelsize=7)
    ax_f.set_xlim(-11, 42)
    ax_f.set_ylim(-3, 53)
    ax_f.set_aspect('equal')
    ax_f.grid(True, alpha=0.2)

    # Parallelogram grid lines
    for r in range(N_ROWS):
        ax_f.plot(TAXEL_Y[r, :], TX_MAX - TAXEL_X[r, :], '-', color='gray',
                  lw=0.4, alpha=0.3, zorder=1)
    for c in range(N_COLS):
        ax_f.plot(TAXEL_Y[:, c], TX_MAX - TAXEL_X[:, c], '-', color='gray',
                  lw=0.4, alpha=0.3, zorder=1)

    # Scatter heatmap (colour = normal force)
    sc = ax_f.scatter(flat_x, flat_y,
                      c=np.zeros(N_ROWS * N_COLS),
                      cmap='YlOrRd', vmin=0, vmax=VMAX_FN,
                      s=60, zorder=3, edgecolors='none')
    heatmaps.append(sc)

    # Quiver for tangential forces (in flat physical space)
    # Convert (ft1=row-axis, ft2=col-axis) tangential to physical directions
    # ft1 is along the circumferential direction, ft2 along axial
    # In the helix-corrected frame the arrows point in the local PCB axes
    qv = ax_f.quiver(flat_x, flat_y,
                     np.zeros(N_ROWS * N_COLS),
                     np.zeros(N_ROWS * N_COLS),
                     color='steelblue', alpha=0.8,
                     scale=1500, scale_units='xy',
                     width=0.003, zorder=4)
    quivers.append(qv)

    flat_axes.append(ax_f)

    # ── 3D cylinder ────────────────────────────────────────────────────────────
    ax_c = fig.add_subplot(gs[1, i], projection='3d')
    ax_c.set_title(FINGER_NAMES[i], fontsize=9)
    ax_c.tick_params(labelsize=6)
    ax_c.set_xlabel('X', fontsize=6, labelpad=1)
    ax_c.set_ylabel('Y', fontsize=6, labelpad=1)
    ax_c.set_zlabel('Z', fontsize=6, labelpad=1)

    # Draw a faint wire cylinder outline for context
    theta_wire = np.linspace(np.radians(-135), np.radians(135), 40)
    z_wire = np.array([0, 0.05])
    for zw in z_wire:
        ax_c.plot(A_SEMI * np.cos(theta_wire),
                  B_SEMI * np.sin(theta_wire),
                  np.full_like(theta_wire, zw),
                  color='lightgray', lw=0.6, alpha=0.4)

    # Scatter: one dot per taxel, colour = normal force
    xs = TAXEL_3D[:, :, 0].flatten()
    ys = TAXEL_3D[:, :, 1].flatten()
    zs = TAXEL_3D[:, :, 2].flatten()
    sc3 = ax_c.scatter(xs, ys, zs,
                       c=np.zeros(N_ROWS * N_COLS),
                       cmap='YlOrRd', vmin=0, vmax=VMAX_FN,
                       s=25, depthshade=False, zorder=3)
    surf_plots.append(sc3)

    # 3D quiver for tangential forces
    qv3 = ax_c.quiver(xs, ys, zs,
                      np.zeros(N_ROWS * N_COLS),
                      np.zeros(N_ROWS * N_COLS),
                      np.zeros(N_ROWS * N_COLS),
                      length=0.003, normalize=False,
                      color='steelblue', alpha=0.7)
    quiv3d.append(qv3)

    ax_c.set_xlim(-0.013, 0.013)
    ax_c.set_ylim(-0.013, 0.013)
    ax_c.set_zlim(-0.005, 0.055)
    ax_c.view_init(elev=25, azim=-60)
    cyl_axes.append(ax_c)

fig.colorbar(heatmaps[0], ax=flat_axes, label='Normal force Fn',
             fraction=0.01, pad=0.02, shrink=0.6)

# ── Tangential force → 3D direction ───────────────────────────────────────────
# The two tangential components ft1, ft2 are in the sensor's local PCB frame.
# We need unit vectors for the PCB row-direction and col-direction in 3D space.

def build_local_axes():
    """
    For each taxel, compute the unit vectors in the row (+circumferential)
    and col (+axial/distal) directions in 3D, accounting for the helix.
    Returns d_row, d_col each shaped (N_ROWS*N_COLS, 3).
    """
    d_row = np.zeros((N_ROWS, N_COLS, 3))
    d_col = np.zeros((N_ROWS, N_COLS, 3))

    for r in range(N_ROWS):
        for c in range(N_COLS):
            # Finite difference for row direction
            r1 = min(r + 1, N_ROWS - 1)
            r0 = max(r - 1, 0)
            dr = TAXEL_3D[r1, c] - TAXEL_3D[r0, c]
            d_row[r, c] = dr / (np.linalg.norm(dr) + 1e-10)

            # Finite difference for col direction
            c1 = min(c + 1, N_COLS - 1)
            c0 = max(c - 1, 0)
            dc = TAXEL_3D[r, c1] - TAXEL_3D[r, c0]
            d_col[r, c] = dc / (np.linalg.norm(dc) + 1e-10)

    return d_row.reshape(-1, 3), d_col.reshape(-1, 3)

D_ROW, D_COL = build_local_axes()

# ── Data ───────────────────────────────────────────────────────────────────────
data_cache = [np.zeros((N_ROWS, N_COLS, 3)) for _ in range(5)]

def _state_callback(state_msg):
    global data_cache
    state = XHandState.from_msg(state_msg)
    for i, sensor in enumerate(state.sensor.force_raw):
        data_cache[i] = sensor.astype(float)

rospy.init_node('xhand_sensor_feedback')
rospy.Subscriber("/xhand_control/xhand_state", XHandStateArrayMsg, _state_callback)

# ── Main loop ─────────────────────────────────────────────────────────────────
try:
    while True:
        for i, data in enumerate(data_cache):
            ft1 = data[:, :, 0].flatten()   # circumferential tangential
            ft2 = data[:, :, 1].flatten()   # axial tangential
            fn  = data[:, :, 2].flatten()   # normal

            # Flat unwrap
            heatmaps[i].set_array(fn)
            # Tangential in flat physical coords:
            # ft1 acts in the circ (y) direction, ft2 in axial (x) — both helix-sheared
            # Approximate: use raw ft1→dy, ft2→dx in the flat parallelogram
            quivers[i].set_UVC(ft1, -ft2)   # x=circ=ft1, y=axial(flipped)=-ft2

            # 3D cylinder
            surf_plots[i].set_array(fn)

            # Rebuild 3D quiver (matplotlib requires remove+re-add)
            quiv3d[i].remove()
            xs = TAXEL_3D[:, :, 0].flatten()
            ys = TAXEL_3D[:, :, 1].flatten()
            zs = TAXEL_3D[:, :, 2].flatten()
            scale = 2e-4  # tune: scales force units → metres
            uvw = ft1[:, None] * D_ROW + ft2[:, None] * D_COL
            quiv3d[i] = cyl_axes[i].quiver(
                xs, ys, zs,
                uvw[:, 0] * scale,
                uvw[:, 1] * scale,
                uvw[:, 2] * scale,
                color='steelblue', alpha=0.7,
                length=1, normalize=False
            )

        plt.draw()
        plt.pause(0.1)

except KeyboardInterrupt:
    pass
finally:
    plt.ioff()
    plt.show()