from vpython import *
import requests
import math
import numpy as np

URL = "http://172.20.10.1"

scene = canvas(title="Smartphone Motion | Attitude sensor(RPW) ",
               width=900, height=500,
               background=color.white)

scene.camera.pos  = vector(5,5,10)
scene.camera.axis = vector(-5,-5,-10)

phone = box(size=vector(2.8,0.3,1.5), color=color.cyan)

top_marker = box(pos=vector(0,0.2,-0.6),
                 size=vector(0.4,0.15,0.3),
                 color=color.red)

axis_len = 2
axis_thick = 0.02


label(pos=vector(0,0,0),
      text="O",
      box=False,
      height=12,
      color=color.black)

# X 
arrow(pos=vector(0,0,0),
      axis=vector(-1.2,-2,0),
      color=color.orange,
      shaftwidth=axis_thick)

label(pos=vector(-1.5,-2.4,0),
      text="X",
      box=False,
      height=14,
      color=color.orange)

# Y → 
arrow(pos=vector(0,0,0),
      axis=vector(axis_len,0,0),
      color=color.green,
      shaftwidth=axis_thick)

label(pos=vector(axis_len+0.2,0,0),
      text="Y",
      box=False,
      height=14,
      color=color.green)

# Z ↑ 
arrow(pos=vector(0,0,0),
      axis=vector(0,axis_len,0),
      color=color.blue,
      shaftwidth=axis_thick)

label(pos=vector(0,axis_len+0.3,0),
      text="Z",
      box=False,
      height=14,
      color=color.blue)



# ── LABEL ──────────────────────────────
info_lbl = label(pos=vector(0,6,0), text="Waiting...", box=False)
pan_lbl  = label(pos=vector(0,5,0), text="", box=False, color=color.blue)
tilt_lbl = label(pos=vector(0,4.5,0), text="", box=False, color=color.green)
roll_lbl = label(pos=vector(0,4,0), text="", box=False, color=color.orange)

raw_lbl = label(pos=vector(0,5.5,0),
                text="RAW: --",
                box=False,
                color=color.gray(0.3),
                height=11)

fft_lbl = label(pos=vector(-5,1,0),
                text="FFT: --",
                box=False,
                color=color.black,
                height=12)



# ── READ DATA ──────────────────────────
def read_data():
    try:
        r = requests.get(f"{URL}/get?w&x&y&z&pitch&roll&yaw", timeout=1)
        data = r.json()
        buf = data["buffer"]

        qw = buf["w"]["buffer"][-1]
        qx = buf["x"]["buffer"][-1]
        qy = buf["y"]["buffer"][-1]
        qz = buf["z"]["buffer"][-1]

        pitch = buf["pitch"]["buffer"][-1]
        roll  = buf["roll"]["buffer"][-1]
        yaw   = buf["yaw"]["buffer"][-1]

        return qw,qx,qy,qz,pitch,roll,yaw
    except:
        return None

# ── QUATERNION ROTATION ────────────────
def rotate(qw,qx,qy,qz,vx,vy,vz):
    tw = -qx*vx - qy*vy - qz*vz
    tx =  qw*vx + qy*vz - qz*vy
    ty =  qw*vy - qx*vz + qz*vx
    tz =  qw*vz + qx*vy - qy*vx

    rx = tx*qw - tw*qx - ty*qz + tz*qy
    ry = ty*qw - tw*qy + tx*qz - tz*qx
    rz = tz*qw - tw*qz - tx*qy + ty*qx

    return rx,ry,rz


# ── FFT BUFFER ─────────────────────────
buffer_size = 128
pitch_buffer = []
roll_buffer  = []
yaw_buffer   = []
dt = 1/30



# ── MAIN LOOP ─────────────────────────
while True:
  
    rate(30)

    data = read_data()
    if data is None:
        continue

    qw,qx,qy,qz,pitch,roll,yaw = data

    # ── FIX MAPPING ──
    real_roll  = pitch
    real_pitch = yaw
    real_yaw   = roll

    # ── STORE DATA ──
    pitch_buffer.append(real_pitch)
    roll_buffer.append(real_roll)
    yaw_buffer.append(real_yaw)

    # keep buffer size
    if len(pitch_buffer) > buffer_size:
        pitch_buffer.pop(0)
        roll_buffer.pop(0)
        yaw_buffer.pop(0)

    # ── FFT ──
    if len(pitch_buffer) == buffer_size:
        data_fft = np.array(pitch_buffer)

        # remove DC
        data_fft = data_fft - np.mean(data_fft)

        fft_vals = np.fft.fft(data_fft)
        freqs = np.fft.fftfreq(len(data_fft), dt)

        # เอาเฉพาะ positive freq
        idx = np.where(freqs > 0)

        freqs = freqs[idx]
        mags  = np.abs(fft_vals[idx])

        # dominant frequency
        peak_freq = freqs[np.argmax(mags)]

        fft_lbl.text = f"FFT Peak: {peak_freq:.2f} Hz"


    # ── ROTATE OBJECT ──
    ax,ay,az = rotate(qw,qx,qy,qz,1,0,0)
    ux,uy,uz = rotate(qw,qx,qy,qz,0,1,0)

    phone.axis = vector(ax,ay,az)
    phone.up   = vector(ux,uy,uz)

    mx,my,mz = rotate(qw,qx,qy,qz,0,0.2,-0.6)
    top_marker.pos  = vector(mx,my,mz)
    top_marker.axis = vector(ax,ay,az)
    top_marker.up   = vector(ux,uy,uz)

    # ── SERVO ──
    pan  = max(-90, min(90, real_yaw))
    tilt = max(-45, min(45, real_pitch))

    servo1 = (pan + 90)
    servo2 = (tilt + 45)

    # ── LABEL ──
    pan_lbl.text  = f"Pan (Yaw): {pan:+.1f}°"
    tilt_lbl.text = f"Tilt (Pitch): {tilt:+.1f}°"
    roll_lbl.text = f"Roll: {real_roll:+.1f}°"
    info_lbl.text = f"Pitch:{real_pitch:.1f}  Roll:{real_roll:.1f}  Yaw:{real_yaw:.1f}"
    raw_lbl.text = f"RAW ==> Pitch:{pitch:.1f}  Roll:{roll:.1f}  Yaw:{yaw:.1f}"