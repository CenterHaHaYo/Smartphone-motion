# ============================================================
# Pan-Tilt 2-DOF Visualization
# Sensor : Acceleration with g (accX, accY, accZ)
# Pan    = Yaw   → BUT accelerometer cannot give yaw
#                  so Pan follows ROLL instead
#                  (tilting phone left/right = pan left/right)
# Tilt   = Pitch → tilt phone forward/back = tilt up/down
# One box only, with direction arrows
# ============================================================


# code number 2
from vpython import *
import requests
import math

# ── STEP 1: Create 3D window ──────────────────────────────────
scene = canvas(
    title      = "Pan-Tilt 2-DOF  |  Accelerometer",
    width      = 900,
    height     = 600,
    background = color.white
)
scene.camera.pos  = vector(0, 5, 12)
scene.camera.axis = vector(0, -3, -10)

# ── STEP 2: One moving box (the camera block) ─────────────────
cam_box = box(
    pos   = vector(0, -1.5, 0),
    size  = vector(2, 0.3, 2),
    color = color.cyan
)


# ── STEP 4: Pan arrows (LEFT / RIGHT) ─────────────────────────
# Blue arrows → show Pan direction (horizontal rotation)
arrow(pos=vector( 0, -4, 0), axis=vector( 3.5, 0, 0),
      color=color.blue, shaftwidth=0.08)
arrow(pos=vector( 0, -4, 0), axis=vector(-3.5, 0, 0),
      color=color.blue, shaftwidth=0.08)

# ── STEP 5: Tilt arrows (UP / DOWN) ───────────────────────────
# Red arrows → show Tilt direction (vertical rotation)
arrow(pos=vector(-3.5, 0, 0), axis=vector(0,  3, 0),
      color=color.green, shaftwidth=0.08)
arrow(pos=vector(-3.5, 0, 0), axis=vector(0, -3, 0),
      color=color.green, shaftwidth=0.08)


# ── Axis arrows (reference frame) ────────────────────────────────────────────
arrow(pos=vector(0,-1.5,0), axis=vector(2,0,0), color=color.red,   shaftwidth=0.05)
arrow(pos=vector(0,-1.5,0), axis=vector(0,2,0), color=color.green, shaftwidth=0.05)
arrow(pos=vector(0,-1.5,0), axis=vector(0,0,2), color=color.blue,  shaftwidth=0.05)
label(pos=vector(2.2,-1.5,0),   text="X", box=False, color=color.red)
label(pos=vector(0, 0.7,0),     text="Z", box=False, color=color.green)
label(pos=vector(0,-1.5,2.2),   text="Y", box=False, color=color.blue)


# ── STEP 6: Direction labels ───────────────────────────────────
label(pos=vector( 4.5, -4, 0),
      text="Pan left/right",  box=False,
      color=color.blue, height=13)
label(pos=vector(-3.5,   3.5, 0),
      text="Tilt up/down",    box=False,
      color=color.green,  height=13)

# ── STEP 7: Live data labels ───────────────────────────────────
info_lbl = label(pos=vector(0, 5.5, 0),
                 text="Waiting for data...",
                 box=False, height=14)

pan_label  = label(pos=vector(-7, 1, 0),
                   text="Pan  (Yaw):  0.0°",
                   box=False, height=12,
                   color=color.blue)

tilt_label = label(pos=vector(-7, 0, 0),
                   text="Tilt (Pitch): 0.0°",
                   box=False, height=12,
                   color=color.green)

servo_lbl = label(pos=vector(0, 3.2, 0),
                  text="Servo Pan: 90°  Servo Tilt: 90°",
                  box=False, color=color.gray(0.3), height=12)


URL = "http://172.20.10.1"   

# ── STEP 9: Read accelerometer from Phyphox ───────────────────
def read_accel():
    try:
        r = requests.get(
            f"{URL}/get?accX&accY&accZ",
            timeout = 1
        )
        data = r.json()

        # Check phone is recording
        if not data["status"]["measuring"]:
            print("Press PLAY in Phyphox!")
            return None

        buf = data["buffer"]
        ax  = buf["accX"]["buffer"][0]
        ay  = buf["accY"]["buffer"][0]
        az  = buf["accZ"]["buffer"][0]

        if any(v is None for v in (ax, ay, az)):
            print("Waiting for sensor data...")
            return None

        return float(ax), float(ay), float(az)

    except Exception as e:
        print(f"[ERROR] {e}")
        return None

# ── STEP 10: Clamp helper ──────────────────────────────────────
def clamp(val, lo, hi):
    return max(lo, min(hi, val))

# ── STEP 11: Main loop ─────────────────────────────────────────
while True:
    rate(30)   # 30 updates per second

    accel = read_accel()
    if accel is None:
        continue

    ax, ay, az = accel

    # ── HOW WE GET PAN AND TILT FROM ACCELEROMETER ────────────
    #
    # Accelerometer measures gravity direction
    # We use the gravity vector to calculate two angles:
    #
    # TILT (Pitch) = how much phone tilts FORWARD / BACKWARD
    #   formula: atan2(ay, sqrt(ax²+az²))
    #   phone flat     → pitch =   0°
    #   phone face up  → pitch = +90°
    #   phone face down→ pitch = -90°
    #
    # PAN (Roll)   = how much phone tilts LEFT / RIGHT
    #   formula: atan2(-ax, az)
    #   phone flat     → roll  =   0°
    #   phone left     → roll  = -90°
    #   phone right    → roll  = +90°
    #
    # NOTE: True Yaw (spinning phone flat on table) is NOT
    # possible with accelerometer only.
    # Roll is the closest we can do for Pan.

    pitch_rad = math.atan2(ay, math.sqrt(ax**2 + az**2))
    roll_rad  = math.atan2(-ax, az)

    # ── PAN-TILT MAPPING (from your document) ─────────────────
    # Pan  = Yaw   → we use Roll  (best we can do)
    # Tilt = Pitch → we use Pitch (perfect)
    pan_rad  = roll_rad
    tilt_rad = pitch_rad

    pan_deg  = math.degrees(pan_rad)
    tilt_deg = math.degrees(tilt_rad)

    # ── CLAMP TO SERVO RANGES ──────────────────────────────────
    # Pan  : -90° to +90°
    # Tilt : -45° to +45°
    pan_deg  = clamp(pan_deg,  -90, 90)
    tilt_deg = clamp(tilt_deg, -45, 45)

    pan_rad  = math.radians(pan_deg)
    tilt_rad = math.radians(tilt_deg)

    # ── SERVO ANGLE MAPPING ────────────────────────────────────
    # Pan  servo: -90° → 0°,   0° → 90°,  +90° → 180°
    # Tilt servo: -45° → 30°,  0° → 90°,  +45° → 150°
    servo_pan  = (pan_deg  + 90) / 180 * 180
    servo_tilt = (tilt_deg + 45) / 90  * 120 + 30

    # ── ROTATE THE BOX ─────────────────────────────────────────
    #
    # Pan  = rotation around Z axis (vertical)
    #        → box turns LEFT or RIGHT
    #
    # Tilt = rotation around Y axis (horizontal)
    #        → box turns UP or DOWN
    #
    # Forward direction of box after Pan rotation:
    forward = vector(
        math.cos(pan_rad),    # X
        0,                    # Y  (pan stays flat)
       -math.sin(pan_rad)     # Z
    )

    # Up direction of box after Tilt rotation:
    up_vec = vector(
        math.sin(pan_rad)  * math.sin(tilt_rad),   # X
        math.cos(tilt_rad),                         # Y
        math.cos(pan_rad)  * math.sin(tilt_rad)    # Z
    )

    # Apply to box
    cam_box.axis = forward
    cam_box.up   = up_vec

    # ── UPDATE LABELS ──────────────────────────────────────────
    pan_label.text   = f"Pan  (Yaw):  {pan_deg:+.1f}°"
    tilt_label.text  = f"Tilt (Pitch): {tilt_deg:+.1f}°"
    servo_lbl.text = (f"Servo Pan: {servo_pan:.0f}°   "
                      f"Servo Tilt: {servo_tilt:.0f}°")
    info_lbl.text  = (f"accX={ax:.2f}  "
                      f"accY={ay:.2f}  "
                      f"accZ={az:.2f}")
    