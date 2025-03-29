#///////////////////////////////////////////////////////////////////////////////
#//     Filter-Adapted Spatio-Temporal Sampling With General Distributions    //
#//        Copyright (c) 2024 Electronic Arts Inc. All rights reserved.       //
#///////////////////////////////////////////////////////////////////////////////

import Host
import GigiArray
import numpy as np
from PIL import Image
import os

ReadbackResource = "ToneMap_Tonemap.SDR: ToneMap_Color_SDR (UAV - After)"
PTConvergeFrames = 500
EMAConvergeFrames = 120

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

# Set up post effect DOF parameters
Host.SetImportedBufferFile("Scene", "..\\Assets\\Exterior\\exterior.bin")
Host.SetImportedBufferFile("VertexBuffer", "..\\Assets\\Exterior\\exterior.bin")
Host.SetVariable("RenderSize", "1024, 786")
Host.SetVariable("MaterialSet", "Exterior")
Host.SetVariable("Accumulate", "true")
Host.SetVariable("Animate", "true")
Host.SetVariable("SamplesPerPixelPerFrame", "8")
Host.SetVariable("JitterPixels", "PerPixel")
Host.SetVariable("NumBounces", "4")  # for foliage. It uses a bounce per alpha test, so a higher bounce count is needed for foliage. Toning it down for post processing fx though
Host.SetVariable("AlbedoMode", "false")
Host.SetVariable("AlbedoModeAlbedoMultiplier", "0.25")
Host.SetVariable("FocalLength", "250")
Host.SetVariable("JitterNoiseTextures", "false")
Host.SetVariable("DOF", "PostProcessing")
Host.SetVariable("ApertureRadius", "5")
Host.SetVariable("AnamorphicScaling", "1,1")
Host.SetVariable("PetzvalScaling", "1,1")
Host.SetVariable("OcclusionSettings", "1,1,1")
Host.SetVariable("NoImportanceSampling", "false")
Host.SetVariable("SkyColor", "1,1,1")
Host.SetVariable("SkyBrightness", "1")
Host.SetVariable("MaterialEmissiveMultiplier", "100")
Host.SetVariable("GatherDOF_AnimateNoiseTextures", "true")
Host.SetVariable("GatherDOF_UseNoiseTextures", "false")
Host.SetVariable("GatherDOF_SuppressBokeh", "false")
Host.SetVariable("GatherDOF_FocalDistance", "500")
Host.SetVariable("GatherDOF_FocalRegion", "100")
Host.SetVariable("GatherDOF_FocalLength", "75")
Host.SetVariable("GatherDOF_NearTransitionRegion", "50")
Host.SetVariable("GatherDOF_FarTransitionRegion", "200")
Host.SetVariable("GatherDOF_Scale", "0.5")
Host.SetVariable("GatherDOF_DoFarField", "true")
Host.SetVariable("GatherDOF_DoFarFieldFloodFill", "true")
Host.SetVariable("GatherDOF_DoNearField", "true")
Host.SetVariable("GatherDOF_DoNearFieldFloodFill", "true")
Host.SetVariable("GatherDOF_KernelSize", "20, 1.57, 6, 0")
Host.SetVariable("GatherDOF_BlurTapCount", "8")
Host.SetVariable("GatherDOF_FloodFillTapCount", "4")
Host.SetVariable("GaussBlur_Disable", "true")
Host.SetVariable("SmallLightBrightness","200")
Host.SetVariable("SmallLightsColor","1,1,1")
Host.SetVariable("SmallLightsColorful","true")
Host.SetVariable("SmallLightRadius","1")
Host.SetVariable("TemporalAccumulation_Alpha","0.1")
Host.SetVariable("TemporalAccumulation_Enabled","true")
Host.SetVariable("ToneMap_ExposureFStops", "1")
Host.SetVariable("ToneMap_ToneMapper", "ACES")
Host.SetVariable("LensRNGExtend", "Shuffle1D")

Host.RunTechnique()
Host.WaitOnGPU()

# Run the path tracer to a decent convergence so we have something to apply post processing to
# Make sure the PT accumulation buffer, and the temporal accumulation buffer are cleared
Host.Print("Path Tracing To Convergence...")
Host.SetVariable("TemporalAccumulation_Alpha","1.0")
Host.SetVariable("FrameIndex", "0")
Host.SetVariable("Reset", "true")
for i in range(PTConvergeFrames):
	Host.RunTechnique()
Host.SetVariable("TemporalAccumulation_Alpha","0.1")
Host.Print("Path Tracing Converged")

# High Quality. All settings are correct already.
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_HQ.png")

# Very Low Quality.
Host.SetVariable("GatherDOF_DoFarFieldFloodFill", "false")
Host.SetVariable("GatherDOF_DoNearFieldFloodFill", "false")
Host.SetVariable("GatherDOF_BlurTapCount", "4")
Host.SetVariable("TemporalAccumulation_Alpha","1.0")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_LQ.png")
Host.SetVariable("TemporalAccumulation_Alpha","0.1")

# Blue
Host.SetVariable("GatherDOF_UseNoiseTextures", "true")
Host.SetVariable("LensRNGSource", "UniformHexagonBlue")
Host.SetVariable("TemporalAccumulation_Alpha","1.0")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_Blue_1spp.png")
Host.SetVariable("TemporalAccumulation_Alpha","0.1")
for i in range(EMAConvergeFrames):
	Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_Blue_" + str(EMAConvergeFrames) + "spp.png")

# ICDF Blue
Host.SetVariable("GatherDOF_UseNoiseTextures", "true")
Host.SetVariable("LensRNGSource", "UniformHexagonICDF_Blue")
Host.SetVariable("TemporalAccumulation_Alpha","1.0")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_ICDFBlue_1spp.png")
Host.SetVariable("TemporalAccumulation_Alpha","0.1")
for i in range(EMAConvergeFrames):
	Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_ICDFBlue_" + str(EMAConvergeFrames) + "spp.png")

# White
Host.SetVariable("GatherDOF_UseNoiseTextures", "true")
Host.SetVariable("LensRNGSource", "UniformHexagonWhite")
Host.SetVariable("TemporalAccumulation_Alpha","1.0")
Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_White_1spp.png")
Host.SetVariable("TemporalAccumulation_Alpha","0.1")
for i in range(EMAConvergeFrames):
	Host.RunTechnique()
Host.WaitOnGPU()
lastReadback, success = Host.Readback(ReadbackResource)
lastReadbackNp = np.array(lastReadback)
lastReadbackNp = lastReadbackNp.reshape((lastReadbackNp.shape[1], lastReadbackNp.shape[2], lastReadbackNp.shape[3]))
Image.fromarray(lastReadbackNp, "RGBA").save(Host.GetScriptPath() + "FX_Gather_White_" + str(EMAConvergeFrames) + "spp.png")
