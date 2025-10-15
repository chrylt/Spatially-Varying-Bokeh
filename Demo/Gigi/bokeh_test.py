#///////////////////////////////////////////////////////////////////////////////
#//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
#//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
#///////////////////////////////////////////////////////////////////////////////

import Host
import GigiArray
import numpy as np
from PIL import Image
import os

samplesPerPixel = "20"

ReadbackResource = "ToneMap_Tonemap.SDR: ToneMap_Color_SDR (UAV - After)"

# don't save gguser files during this script execution
Host.DisableGGUserSave(True)

def _init_environment():
    """One-time renderer setup."""
    # Do one execution to ensure everything is initialized
    Host.RunTechnique()

    # We want the tone mapped SDR image as output
    Host.SetWantReadback(ReadbackResource)

    # Set camera
    Host.SetCameraPos(462.217, 126.867, 188.942)
    Host.SetCameraAltitudeAzimuth(-0.249, 1.745)
    Host.SetCameraFOV(28.0)

    # Set up path traced albedo mode parameters
    Host.SetVariable("RenderSize", "400, 300")
    Host.SetVariable("MaterialSet", "Exterior")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")
    Host.SetVariable("SamplesPerPixelPerFrame", samplesPerPixel)
    Host.SetVariable("JitterPixels", "None")
    Host.SetVariable("NumBounces", "16")  # for foliage. It uses a bounce per alpha test, so a higher bounce count is needed for foliage.
    Host.SetVariable("AlbedoMode", "true")
    Host.SetVariable("AlbedoModeAlbedoMultiplier", "0.25")
    Host.SetVariable("FocalLength", "250")
    Host.SetVariable("JitterNoiseTextures", "false")
    Host.SetVariable("DOF", "PathTraced")
    Host.SetVariable("ApertureRadius", "5")
    Host.SetVariable("AnamorphicScaling", "1,1")
    Host.SetVariable("PetzvalScaling", "1,1")
    Host.SetVariable("OcclusionSettings", "1,1,1")
    Host.SetVariable("NoImportanceSampling", "false")
    Host.SetVariable("SkyColor", "1,1,1")
    Host.SetVariable("SkyBrightness", "1")
    Host.SetVariable("MaterialEmissiveMultiplier", "100")
    Host.SetVariable("GaussBlur_Disable", "true")
    Host.SetVariable("SmallLightBrightness","200")
    Host.SetVariable("SmallLightsColor","1,1,1")
    Host.SetVariable("SmallLightsColorful","true")
    Host.SetVariable("SmallLightRadius","1")
    Host.SetVariable("TemporalAccumulation_Enabled","false")
    Host.SetVariable("ToneMap_ExposureFStops", "1")
    Host.SetVariable("ToneMap_ToneMapper", "ACES")
    Host.SetVariable("LensRNGExtend", "Shuffle1D")

    Host.RunTechnique()
    Host.WaitOnGPU()

def _render_config(focal_length: float, aperture_stop: float, runs: int):
    fl_str = f"{focal_length}"
    as_str = f"{aperture_stop}"

    print(f"Start render: fl={fl_str}, as={as_str}, runs={runs}")
    Host.SetFrameIndex(0)
    Host.SetVariable("FrameIndex", "0")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    Host.SetVariable("DOF", "Realistic")
    Host.SetVariable("FocalLength", fl_str)
    Host.SetVariable("HeliosApertureStop", as_str)
    Host.SetVariable("BokehTest", "true")
    Host.SetVariable("SmallLightRadius","2.5")
    Host.SetVariable("SmallLightBrightness","50")

    step = max(1, runs // 10)
    for i in range(1, runs + 1):
        Host.RunTechnique()
        if i % step == 0 or i == runs:
            pct = int(i * 100 / runs)

    lastReadback, success = Host.Readback(ReadbackResource)
    lastReadbackNp = np.array(lastReadback)
    lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
    out_path = os.path.join(Host.GetScriptPath(), f"bokehConfig_fl{fl_str}_as{as_str}_run{runs}.png")
    Image.fromarray(lastReadbackNp, "RGBA").save(out_path)
    print(f"Saved: {out_path}")

    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")


# main execution
_init_environment()

# Sweep settings
focal_lengths = [35, 37.8, 40, 42]
aperture_stops = list(range(0, 7))   # 0..6
runs_per_config = 500000

total = len(focal_lengths) * len(aperture_stops)
idx = 0
for fl in focal_lengths:
    for ap in aperture_stops:
        idx += 1
        print(f"\n[{idx}/{total}] Rendering fl={fl}, ap={ap}")
        _render_config(fl, ap, runs_per_config)


# Exit
#Host.Exit(exitCode)
