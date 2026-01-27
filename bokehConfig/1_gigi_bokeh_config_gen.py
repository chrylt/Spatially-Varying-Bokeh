import Host
import GigiArray
import numpy as np
import OpenEXR
import Imath
import os

# Settings
ReadbackResource = "Raytrace.BokehConfigOut: ColorHDR___ (UAV - After)"
focus_distances = [45.0]
object_distances = [300]
aperture_stops = [6] # 0.. 6
samples_per_pixel_per_frame = 16
sample_count_total = 10000
light_count = 15
light_size = 0.3

runs_per_config =  sample_count_total // samples_per_pixel_per_frame

# don't save gguser files during this script execution
Host.DisableGGUserSave(True)

def init():
    # variables to control bokeh rendering
    Host.SetVariable("SmallLightBrightness","100")
    Host.SetVariable("ToggleChromaticAberration", "false")

    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("BokehConfigMode", "RealisticLens")
    
    # turn off other renderers for better performance
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("RenderThinLensDoF", "false")
    Host.SetVariable("RenderLensSimulationDoF", "false")

    Host.SetVariable("NumBounces", "2") # bounces not necessary for bokeh config

    Host.SetWantReadback(ReadbackResource)

def _render_config(focus_distance: float, aperture_stop: float, object_distance: float, light_index: int = -1):
    fd_str = f"{focus_distance}"
    as_str = f"{aperture_stop}"

    print(f"Rendering: focus_distance={fd_str}, aperture_stop={as_str}, runs={runs_per_config}, light_index={light_index}")
    Host.SetFrameIndex(0)
    Host.SetVariable("FrameIndex", "0")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    Host.SetVariable("FocusDistance", fd_str)
    Host.SetVariable("HeliosApertureStop", as_str)
    Host.SetVariable("OnlyThisLightByIndex", str(light_index))
    
    Host.SetVariable("SmallLightRadius",str(light_size))
    Host.SetVariable("ConfigLightDistance", str(object_distance))
    
    for i in range(0, runs_per_config):
        Host.RunTechnique()

    lastReadback, _ = Host.Readback(ReadbackResource)
    lastReadbackNp = np.array(lastReadback)
    lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
    
    # Extract single grayscale channel (red channel) and keep as 32-bit float HDR
    grayscale = lastReadbackNp[:, :, 0].astype(np.float32)
    
    # Save as EXR with single channel
    height, width = grayscale.shape
    header = OpenEXR.Header(width, height)
    header['channels'] = {'Y': Imath.Channel(Imath.PixelType(Imath.PixelType.FLOAT))}
    
    out_path = os.path.join(Host.GetScriptPath(), f"1_rawRenderings\\bokeh_focus{fd_str}_aperture{as_str}_samples{sample_count_total}_distance{object_distance}_light{light_index+1}of{light_count}.exr")
    exr_file = OpenEXR.OutputFile(out_path, header)
    exr_file.writePixels({'Y': grayscale.tobytes()})
    exr_file.close()
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
                _render_config(fd, ap, object_distance=object_distance, light_index=light)
