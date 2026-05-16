# VLA capture

Records 1080p30 MJPEG video from a UVC webcam into a single session-long
`.mkv` file, with per-frame metadata in `frames.jsonl` and episode boundaries
(spacebar toggles) in `events.jsonl`. Robot state capture (via roslibpy →
`/xhand_control/XHandState`) is stubbed for now.

The camera records continuously across the whole session — episode markers
are just timestamp boundaries in `events.jsonl`. Episodes are extracted in a
post-processing pass by slicing `frames.jsonl` and `robot_state.jsonl` on the
event timestamps.

## Setup

```bash
sudo apt install v4l-utils                 # for camera control
pip install -r requirements.txt
```

## Usage

```bash
# See what controls + formats your specific webcam supports
python capture.py --list-controls

# Camera only (for now, while the robot side is stubbed)
python capture.py --no-robot

# Both (once roslibpy side is implemented)
python capture.py
```

In the preview window: **SPACE** to mark episode start/end, **q** to quit.

## Output

```
recordings/session_YYYYMMDD_HHMMSS/
  rgb.mkv            session-long MJPEG-in-Matroska video
  frames.jsonl       one record per frame
  events.jsonl       one record per episode_start / episode_end
  robot_state.jsonl  one record per ROS sample (TBD)
  session_meta.json  config snapshot, start/end times, totals
```

`frames.jsonl` records:

```json
{"camera":"webcam_rgb","depth_file":null,"frame_index":0,"height":1080,"host_timestamp_ns":1731234567890123456,"rgb_video":"rgb.mkv","rgb_video_frame":0,"width":1920}
```

`events.jsonl` records:

```json
{"event":"episode_start","robot_timestamp_ns":1731234567890123456}
{"event":"episode_end","robot_timestamp_ns":1731234572345678901}
```

Both use `time.time_ns()` (host wall-clock, nanoseconds), so cross-stream
alignment is a straightforward merge on timestamp.

## Notes on the camera setup

- The camera is opened with `MJPG` fourcc — UVC cams (UGreen included) need
  this to deliver 1080p30 over USB; YUYV silently falls back to ~5fps.
- `v4l2-ctl` is used to set autofocus/exposure/white-balance *before* OpenCV
  opens the device. Some controls get rejected once OpenCV has the handle.
- Control names vary slightly across cameras. If a control in `config.yaml`
  doesn't apply, run `--list-controls` and check the exact name your device
  uses (e.g. `exposure_absolute` vs `exposure_time_absolute`).
- Frame grab and disk write run in separate threads with a bounded queue, so
  brief disk hiccups won't drop frames. The grabber warns if the inter-frame
  gap exceeds 1.5× the expected interval.
- Anti-flicker exposure: pick `exposure_time_absolute` as a multiple of the
  mains period — 200 / 300 / 400 (in 100µs units) for UK/EU 50Hz, or
  167 / 250 / 333 for US 60Hz.

## Next steps

1. Plug in the robot. Replace `RobotRecorder` with a roslibpy subscriber to
   `/xhand_control/XHandState` that writes one record per sample to
   `robot_state.jsonl` with a `robot_timestamp_ns` field.
2. Episode trigger. Voice command or fixed-duration timer instead of spacebar.
3. Post-processing: align frames ↔ events ↔ robot samples by `host_timestamp_ns` /
   `robot_timestamp_ns`, slice into episodes, transcode MJPEG → H.264, export
   LeRobot parquet + mp4 chunks.