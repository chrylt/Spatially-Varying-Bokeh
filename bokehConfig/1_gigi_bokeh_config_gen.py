import Host
import os

ReadbackResource = "Raytrace.BokehConfigOut: ColorHDR___ (UAV - After)"
focus_distances = [45.0]
object_distances = [250]
aperture_stops = [6]
samples_per_pixel_per_frame = 16
sample_count_total = 10000 # 1000000
light_count = 12
light_size = 0.3

runs_per_config = sample_count_total // samples_per_pixel_per_frame

Host.DisableGGUserSave(True)

def init():
    Host.SetVariable("SmallLightBrightness", "100")
    Host.SetVariable("ToggleChromaticAberration", "false")
    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("BokehConfigMode", "RealisticLens")
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("RenderThinLensDoF", "false")
    Host.SetVariable("RenderLensSimulationDoF", "false")
    Host.SetVariable("NumBounces", "2")
    Host.SetWantReadback(ReadbackResource)

def _render_config(focus_distance, aperture_stop, object_distance, light_index):
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
    Host.SetVariable("SmallLightRadius", str(light_size))
    Host.SetVariable("ConfigLightDistance", str(object_distance))

    for _ in range(runs_per_config):
        Host.RunTechnique()

    # output path
    out_dir = os.path.join(Host.GetScriptPath(), "1_rawRenderings")
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(
        out_dir,
        f"bokeh_focus{fd_str}_aperture{as_str}_samples{sample_count_total}_distance{object_distance}_light{light_index+1}of{light_count}.exr"
    )

    # write exr
    Host.SaveAsEXR(out_path, ReadbackResource, 0, 0)
    print(f"Saved EXR: {out_path}")

    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

init()

for fd in focus_distances:
    for ap in aperture_stops:
        for light in range(light_count):
            for object_distance in object_distances:
                _render_config(fd, ap, object_distance, light)