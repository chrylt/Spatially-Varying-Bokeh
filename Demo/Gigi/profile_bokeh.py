import Host
import os
import json
from datetime import datetime

# default everything to OFF
Host.SetVariable("RenderBokehConfig", "false")
Host.SetVariable("RenderPinhole", "false")
Host.SetVariable("RenderThinLensDoF", "false")
Host.SetVariable("RenderLensSimulationDoF", "false")
Host.SetVariable("DoGatherDoF", "false")
Host.SetVariable("DoDOFAccumulation", "false")
Host.SetVariable("NumBounces", "2")
Host.SetVariable("ToggleChromaticAberration", "false")
Host.SetVariable("SpatiallyVarying", "false")
Host.SetVariable("FastDistortionGatherDoF", "false")

# Profiling settings
num_warmup_frames = 10
num_profiling_frames = 100

# gigi settings for script profiling
Host.DisableGGUserSave(True)
Host.SetProfilingMode(False)

def letPinholeAccumulateScene():
    # Turn Pinhole on
    Host.SetVariable("NumBounces", "16")
    Host.SetVariable("RenderPinhole", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")

    # todo: reset and run for a good amount of frames, so that pinhole path tracing is converged
    # todo: save rendered pinhole scene image "Raytrace.PinholeOut: ColorHDR (UAV - After)" as exr

    # Turn Pinhole off again
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileGatherDoFSpatiallyConstant():
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "false")

    # todo: run technique and save profiling data labeled appropriately

    Host.SetVariable("DoGatherDoF", "false")


def profileGatherDoFSpatiallyVarying():
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "true")

    # todo: run technique and save profiling data labeled appropriately

    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("SpatiallyVarying", "false")

def generateGroundTruthDataScene():
    Host.SetVariable("RenderLensSimulationDoF", "true")
    Host.SetVariable("NumBounces", "16")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")

    # todo: run technique and save ground truth image "ToneMapLens_Simulation_Tonemap.SDR: ToneMapLens_Simulation_Color_SDR (UAV - After)" and "Raytrace.LensSimulationOut: ColorHDR__ (UAV - After)"
    
    Host.SetVariable("RenderLensSimulationDoF", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")
