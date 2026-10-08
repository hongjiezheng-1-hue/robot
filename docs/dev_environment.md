# Development environment notes

## Local Windows environment used for testing

A virtual environment created from an existing conda Python 3.9.16 (the only available interpreter with working ssl), kept outside the repository:

```
python -m venv F:\envs\robot
F:\envs\robot\Scripts\python.exe -m pip install --no-cache-dir --only-binary=:all: mujoco opencv-contrib-python numpy scipy matplotlib
```

Installed versions: mujoco 3.3.7, opencv-contrib-python 5.0.0.93, numpy 2.0.2, scipy 1.13.1, matplotlib 3.9.4. `--only-binary=:all:` is needed because without it pip tried to build mujoco from source and failed; 3.3.7 is the version it then selected for Python 3.9 on Windows. The Colab notebooks run MuJoCo 3.15.0, so the two setups are not identical (see the findings log).

Commands (from the repository root, with the fix below on `PYTHONPATH`):

```
python tests/test_models.py          # also test_ekf, test_replay, test_scene, test_marker_pose, test_wheel_speed_controller, test_mujoco_run_helpers
python experiments/mujoco_ekf.py --model-dir <robotis_mujoco_menagerie>/robotis_tb3 --duration 240
```

## Known problem: `import mujoco` fails with WinError 1114

Symptom: `OSError: [WinError 1114]` while MuJoCo loads its plugin DLLs; `mujoco.dll` itself loads, the four plugin DLLs do not.

Cause (diagnosed by experiment): the base conda environment bundles an old C++ runtime (`msvcp140.dll` 14.27) that the interpreter loads first, while the plugins need a newer one (the system copy is 14.50). Loading the system runtime before `import mujoco` makes the import work.

Fix, kept outside the repository as a `sitecustomize.py` in a folder on `PYTHONPATH` (Python imports it at start-up):

```python
import os, sys
if sys.platform == "win32":
    import ctypes
    system32 = os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32")
    for name in ("msvcp140.dll", "msvcp140_1.dll", "msvcp140_2.dll", "vcruntime140_1.dll", "concrt140.dll"):
        path = os.path.join(system32, name)
        if os.path.exists(path):
            try:
                ctypes.WinDLL(path)
            except OSError:
                pass
```

A standard python.org interpreter does not need this.

## Marker texture on a box depends on the MuJoCo version

With MuJoCo 3.3.7 a 2D texture on a box geom renders as a plain grey face (the texture data is loaded correctly but not mapped), so the marker is not detected; a cube texture built from the same image renders the marker and is detected. MuJoCo 3.15.0 on Colab renders the 2D texture correctly. `SceneConfig.marker_texture_type` selects the type ("2d" by default), and `sim.mujoco_run.pick_texture_type` tries "2d" and then "cube" and keeps the first that is detected in a test render.

An early diagnostic misled: the pixel standard deviation of the image centre was 92 for the plain grey face because the crop included black background. Detection and a look at the image were the reliable checks.

## Cross-check between Colab and local

The same robot pose (1.05, 0, 0) gives a marker position error of about 4.2 mm and a heading error of about 0.025 degrees on Colab (MuJoCo 3.15.0) and 4.3 mm and 0.03 degrees locally (MuJoCo 3.3.7, cube texture), so rendering, detection and pose estimation agree. Local rendering takes about 40 ms per frame (Colab about 330 ms) and a full 240 s collection about 56 s.
