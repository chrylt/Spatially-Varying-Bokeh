import Host
import os

# Turn off rendering
Host.SetVariable("RenderBokehConfig", "false")
Host.SetVariable("RenderPinhole", "false")
Host.SetVariable("RenderThinLensDoF", "false")
Host.SetVariable("RenderLensSimulationDoF", "false")

# Turn off gatherDoFs
Host.SetVariable("DoGatherDoF", "false")
Host.SetVariable("DoDOFAccumulation", "false")

# other performance-impacting settings
Host.SetVariable("NumBounces", "2")
Host.SetVariable("ToggleChromaticAberration", "false")


