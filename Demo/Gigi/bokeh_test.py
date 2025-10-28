import Host
import GigiArray
import numpy as np
from PIL import Image
import os

# Settings
ReadbackResource = "Raytrace.BokehConfigOut: ColorHDR___ (UAV - After)"
focus_distances = [45.0]
object_distances = [500]
aperture_stops = [6] # 0.. 6
samples_per_pixel_per_frame = 16
sample_count_total = 500000
light_count = 8

runs_per_config =  sample_count_total // samples_per_pixel_per_frame

# don't save gguser files during this script execution
Host.DisableGGUserSave(True)

def init():
    # variables to control bokeh rendering
    
    Host.SetVariable("SmallLightBrightness","100")
    Host.SetVariable("ToggleChromaticAberration", "false")
    Host.SetVariable("ConfigLightFieldWidth", "30.400")

    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("BokehConfigMode", "RealisticLens")
    
    # turn off other renderers for better performance
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("RenderThinLensDoF", "false")
    Host.SetVariable("RenderLensSimulationDoF", "false")
    Host.SetVariable("DebugToggle", "false")

    Host.SetVariable("NumBounces", "2") # bounces not necessary for bokeh config

    Host.SetWantReadback(ReadbackResource)

def _render_config(focus_distance: float, aperture_stop: float, object_distance: float, light_index: int = -1):
    fd_str = f"{focus_distance}"
    as_str = f"{aperture_stop}"

    print(f"Start render: fd={fd_str}, as={as_str}, runs={runs_per_config}, light={light_index}")
    Host.SetFrameIndex(0)
    Host.SetVariable("FrameIndex", "0")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    Host.SetVariable("FocusDistance", fd_str)
    Host.SetVariable("HeliosApertureStop", as_str)
    Host.SetVariable("OnlyThisLightByIndex", str(light_index))
    
    Host.SetVariable("SmallLightRadius",str(object_distance / 500)) # take a fraction to keep intensity consistent
    Host.SetVariable("ConfigLightDistance", str(object_distance))
    
    for i in range(0, runs_per_config):
        Host.RunTechnique()

    lastReadback, _ = Host.Readback(ReadbackResource)
    lastReadbackNp = np.array(lastReadback)
    lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
    # clamp 32-bit float HDR data to LDR and pack into 8-bit RGBA for PNG output
    lastReadbackNp = np.clip(lastReadbackNp, 0.0, 1.0)
    lastReadbackNp = (lastReadbackNp * 255.0).astype(np.uint8)
    out_path = os.path.join(Host.GetScriptPath(), f"bokehConfig\\bokeh_fl{fd_str}_as{as_str}_samples{sample_count_total}_od{object_distance}_lidx{light_index}of{light_count}.png")
    Image.fromarray(lastReadbackNp, "RGBA").save(out_path)
    print(f"Saved: {out_path}")

    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

init()

total = len(focus_distances) * len(aperture_stops) * light_count
idx = 0
for fd in focus_distances:
    for ap in aperture_stops:
        for light in range(light_count):
            for object_distance in object_distances:
                idx += 1
                print(f"\n[{idx}/{total}] Rendering fd={fd}, ap={ap}, light={light} / {light_count-1}, od={object_distance}")
                _render_config(fd, ap, object_distance=object_distance, light_index=light)
