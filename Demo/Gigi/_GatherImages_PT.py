#///////////////////////////////////////////////////////////////////////////////
#//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
#//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
#///////////////////////////////////////////////////////////////////////////////

import Host
import GigiArray
import numpy as np
from PIL import Image
import os

samplesPerPixel = "8"

ReadbackResource = "ToneMap_Tonemap.SDR: ToneMap_Color_SDR (UAV - After)"

# don't save gguser files during this script execution
Host.DisableGGUserSave(True)

# Do one execution to ensure everything is initialized
Host.RunTechnique()

# We want the tone mapped SDR image as output
Host.SetWantReadback(ReadbackResource)

# Set camera
Host.SetCameraPos(-506.993, 90.369, 262.885)
Host.SetCameraAltitudeAzimuth(-0.065, 0.060)
Host.SetCameraFOV(45.0)

# Set up path traced albedo mode parameters
Host.SetImportedBufferFile("Scene", "..\\Assets\\Exterior\\exterior.bin")
Host.SetImportedBufferFile("VertexBuffer", "..\\Assets\\Exterior\\exterior.bin")
Host.SetVariable("RenderSize", "1024, 786")
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

# Uniform Star Blue
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarBlue")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_Blue.png")

# Uniform Star White
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarWhite")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_White.png")

# Uniform Star ICDF Blue
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarICDF_Blue")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_ICDFBlue.png")

# Uniform Star ICDF White
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarICDF_White")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_ICDFWhite.png")

# No importance sampling - Blue
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarBlue")
Host.SetVariable("NoImportanceSampling", "true")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_NOISBlue.png")
Host.SetVariable("NoImportanceSampling", "false")

# No importance sampling - White
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarWhite")
Host.SetVariable("NoImportanceSampling", "true")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_" + samplesPerPixel + "spp_NOISWhite.png")
Host.SetVariable("NoImportanceSampling", "false")

# Converged
Host.SetFrameIndex(0)
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("LensRNGSource", "UniformStarWhite")
Host.SetVariable("Accumulate", "true")
Host.SetVariable("Animate", "true")
for i in range(1000):
    Host.RunTechnique()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "PT_AlbedoMode_Converged.png")
Host.SetVariable("Accumulate", "false")
Host.SetVariable("Animate", "false")

# Exit
#Host.Exit(exitCode)
