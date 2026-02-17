import Host
import GigiArray
import numpy as np
from PIL import Image
import os
import json
from datetime import datetime

# Configuration

# Profiling settings
NUM_WARMUP_FRAMES = 10
NUM_PROFILING_FRAMES = 100

# Accumulation settings
PINHOLE_CONVERGE_FRAMES = 500       # Frames to converge pinhole path tracing
DOF_ACCUMULATION_FRAMES = 200       # Frames to accumulate DOF effect
GROUND_TRUTH_CONVERGE_FRAMES = 1000 # Frames for ground truth rendering

# Tap count range for noise level comparison
TAP_COUNT_VALUES = [1, 5, 10, 20, 40, 60, 80]

# Output directory structure
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BASE_OUTPUT_DIR = os.path.join(Host.GetScriptPath(), "profiling_results", TIMESTAMP)

# Resource names for readback
RESOURCE_PINHOLE_HDR = "Raytrace.PinholeOut: ColorHDR (UAV - After)"
RESOURCE_PINHOLE_SDR = "ToneMap_PinholeDOF_Tonemap.SDR: ToneMap_PinholeDOF_Color_SDR (UAV - After)"
RESOURCE_BOKEH_CONFIG_SDR = "ToneMap_BokehConfig_Tonemap.SDR: ToneMap_BokehConfig_Color_SDR (UAV - After)"
RESOURCE_LENS_SIM_SDR = "ToneMapLens_Simulation_Tonemap.SDR: ToneMapLens_Simulation_Color_SDR (UAV - After)"
RESOURCE_LENS_SIM_HDR = "Raytrace.LensSimulationOut: ColorHDR__ (UAV - After)"
RESOURCE_BOKEH_CONFIG_LENS_SDR = "ToneMapLens_BokehConfig_Simulation_Tonemap.SDR: ToneMapLens_BokehConfig_Simulation_Color_SDR (UAV - After)"
RESOURCE_BOKEH_CONFIG_HDR = "Raytrace.BokehConfigOut: ColorHDR___ (UAV - After)"

# Gigi settings for script profiling

Host.DisableGGUserSave(True)
Host.SetProfilingMode(True)
Host.ForceEnableProfiling(True)

# Helper Functions

def create_output_dirs():
    """Create all necessary output directories."""
    dirs = [
        os.path.join(BASE_OUTPUT_DIR, "scene", "profiling"),
        os.path.join(BASE_OUTPUT_DIR, "scene", "images"),
        os.path.join(BASE_OUTPUT_DIR, "scene", "ground_truth"),
        os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "profiling"),
        os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "images"),
        os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "ground_truth"),
        os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "tap_count_comparison"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    Host.Print(f"Output directories created at: {BASE_OUTPUT_DIR}")


def reset_all_variables():
    """Reset all rendering variables to default OFF state."""
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
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")


def reset_accumulation():
    """Reset frame index and accumulation state."""
    Host.SetFrameIndex(0)
    Host.SetVariable("FrameIndex", "0")


def run_profiling(label, num_warmup=NUM_WARMUP_FRAMES, num_frames=NUM_PROFILING_FRAMES):
    """
    Run profiling and return collected data.
    
    Args:
        label: Label for the profiling run
        num_warmup: Number of warmup frames
        num_frames: Number of frames to profile
        
    Returns:
        dict: Profiling data with settings, frames, and summary
    """
    Host.Print(f"Starting profiling: {label}")
    
    # Enable profiling
    Host.SetProfilingMode(True)
    Host.ForceEnableProfiling(True)
    
    # Warmup frames
    Host.Print(f"  Running {num_warmup} warmup frames...")
    Host.RunTechnique(num_warmup)
    Host.WaitOnGPU()
    
    # Collect profiling data
    Host.Print(f"  Profiling {num_frames} frames...")
    frames_data = []
    for i in range(num_frames):
        Host.RunTechnique()
        data = Host.GetProfilingData()
        frames_data.append({
            "frame": i,
            "timings": data
        })
    
    Host.WaitOnGPU()
    
    # Disable profiling
    Host.SetProfilingMode(False)
    Host.ForceEnableProfiling(False)
    
    # Calculate summary statistics
    summary = calculate_profiling_summary(frames_data)
    
    result = {
        "label": label,
        "settings": {
            "warmup_frames": num_warmup,
            "profiling_frames": num_frames
        },
        "frames": frames_data,
        "summary": summary
    }
    
    Host.Print(f"  Profiling complete. Total GPU avg: {summary.get('Total', {}).get('gpu_avg_ms', 'N/A'):.3f} ms")
    return result


def calculate_profiling_summary(frames_data):
    """Calculate min/max/avg statistics for all profiled passes."""
    if not frames_data:
        return {}
    
    # Collect all pass names
    all_passes = set()
    for frame in frames_data:
        all_passes.update(frame["timings"].keys())
    
    summary = {}
    for pass_name in all_passes:
        gpu_times = []
        cpu_times = []
        for frame in frames_data:
            if pass_name in frame["timings"]:
                timing = frame["timings"][pass_name]
                if isinstance(timing, (list, tuple)) and len(timing) >= 2:
                    cpu_times.append(timing[0])
                    gpu_times.append(timing[1])
                elif isinstance(timing, dict):
                    cpu_times.append(timing.get("cpu_ms", 0))
                    gpu_times.append(timing.get("gpu_ms", 0))
        
        if gpu_times:
            summary[pass_name] = {
                "gpu_avg_ms": sum(gpu_times) / len(gpu_times),
                "gpu_min_ms": min(gpu_times),
                "gpu_max_ms": max(gpu_times),
                "cpu_avg_ms": sum(cpu_times) / len(cpu_times) if cpu_times else 0,
                "cpu_min_ms": min(cpu_times) if cpu_times else 0,
                "cpu_max_ms": max(cpu_times) if cpu_times else 0
            }
    
    return summary


def save_profiling_data(profiling_result, output_dir, filename_prefix):
    """Save profiling data to JSON and summary text files."""
    # Save detailed JSON
    json_path = os.path.join(output_dir, f"{filename_prefix}_detail.json")
    with open(json_path, "w") as f:
        json.dump(profiling_result, f, indent=2)
    
    # Save summary text
    summary_path = os.path.join(output_dir, f"{filename_prefix}_summary.txt")
    with open(summary_path, "w") as f:
        f.write(f"Profiling Summary: {profiling_result['label']}\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Warmup frames: {profiling_result['settings']['warmup_frames']}\n")
        f.write(f"Profiling frames: {profiling_result['settings']['profiling_frames']}\n\n")
        f.write("GPU Timings (ms):\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'Pass':<40} {'Avg':>8} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 60 + "\n")
        
        # Sort by average GPU time descending
        summary = profiling_result.get("summary", {})
        sorted_passes = sorted(summary.items(), 
                               key=lambda x: x[1].get("gpu_avg_ms", 0), 
                               reverse=True)
        
        for pass_name, stats in sorted_passes:
            f.write(f"{pass_name:<40} {stats['gpu_avg_ms']:>8.3f} "
                    f"{stats['gpu_min_ms']:>8.3f} {stats['gpu_max_ms']:>8.3f}\n")
    
    Host.Print(f"  Saved profiling data: {filename_prefix}")


def save_image_png(resource_name, output_path, label=""):
    """Save a resource as PNG image using Host.Readback."""
    Host.SetWantReadback(resource_name)
    Host.RunTechnique()
    Host.WaitOnGPU()
    
    readback_data, success = Host.Readback(resource_name)
    if success:
        np_data = np.array(readback_data)
        np_data = np_data.reshape((np_data.shape[1], np_data.shape[2], np_data.shape[3]))
        Image.fromarray(np_data, "RGBA").save(output_path)
        Host.Print(f"  Saved PNG: {os.path.basename(output_path)} {label}")
    else:
        Host.Print(f"  Error: Failed to readback {resource_name}")


def save_image_exr(resource_name, output_path, label=""):
    """Save a resource as EXR image using Host.SaveAsEXR."""
    Host.SetWantReadback(resource_name)
    Host.RunTechnique()
    Host.WaitOnGPU()
    Host.SaveAsEXR(output_path, resource_name, 0, 0)
    Host.Print(f"  Saved EXR: {os.path.basename(output_path)} {label}")


def accumulate_frames(num_frames, reset=True):
    """Run technique for multiple frames to accumulate."""
    if reset:
        reset_accumulation()
    for _ in range(num_frames):
        Host.RunTechnique()
    Host.WaitOnGPU()

# Scene Profiling Functions

def letPinholeAccumulateScene():
    """Accumulate pinhole path tracing for the scene until converged."""
    Host.Print("\n=== Pinhole Accumulation (Scene) ===")
    reset_all_variables()
    
    # Turn Pinhole on
    Host.SetVariable("NumBounces", "16")
    Host.SetVariable("RenderPinhole", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    # Reset and run for convergence
    Host.Print(f"  Accumulating {PINHOLE_CONVERGE_FRAMES} frames for convergence...")
    accumulate_frames(PINHOLE_CONVERGE_FRAMES, reset=True)
    
    # Save rendered pinhole scene image as EXR
    output_path = os.path.join(BASE_OUTPUT_DIR, "scene", "images", "pinhole_converged.exr")
    save_image_exr(RESOURCE_PINHOLE_HDR, output_path, "(converged)")
    
    # Save rendered pinhole scene image as SDR
    output_path_sdr = os.path.join(BASE_OUTPUT_DIR, "scene", "images", "pinhole_converged_sdr.png")
    save_image_png(RESOURCE_PINHOLE_SDR, output_path_sdr, "(converged)")
    
    # Turn Pinhole off again
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileGatherDoFSpatiallyConstantScene():
    """Profile Gather DoF with spatially constant settings for the scene."""
    Host.Print("\n=== Gather DoF Spatially Constant (Scene) ===")
    
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    
    # Run profiling
    profiling_data = run_profiling("GatherDoF_SpatiallyConstant_Scene")
    output_dir = os.path.join(BASE_OUTPUT_DIR, "scene", "profiling")
    save_profiling_data(profiling_data, output_dir, "gather_dof_spatially_constant")
    
    # Accumulate to convergence and save rendered image
    Host.SetVariable("DoDOFAccumulation", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    Host.Print(f"  Accumulating {DOF_ACCUMULATION_FRAMES} frames...")
    accumulate_frames(DOF_ACCUMULATION_FRAMES, reset=True)
    
    output_path = os.path.join(BASE_OUTPUT_DIR, "scene", "images", "gather_dof_spatially_constant.png")
    save_image_png(RESOURCE_PINHOLE_SDR, output_path)
    
    # Reset
    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileGatherDoFSpatiallyVaryingScene():
    """Profile Gather DoF with spatially varying settings for the scene."""
    Host.Print("\n=== Gather DoF Spatially Varying (Scene) ===")
    
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "true")
    Host.SetVariable("DoDOFAccumulation", "false")
    
    # Run profiling
    profiling_data = run_profiling("GatherDoF_SpatiallyVarying_Scene")
    output_dir = os.path.join(BASE_OUTPUT_DIR, "scene", "profiling")
    save_profiling_data(profiling_data, output_dir, "gather_dof_spatially_varying")
    
    # Accumulate to convergence and save rendered image
    Host.SetVariable("DoDOFAccumulation", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    Host.Print(f"  Accumulating {DOF_ACCUMULATION_FRAMES} frames...")
    accumulate_frames(DOF_ACCUMULATION_FRAMES, reset=True)
    
    output_path = os.path.join(BASE_OUTPUT_DIR, "scene", "images", "gather_dof_spatially_varying.png")
    save_image_png(RESOURCE_PINHOLE_SDR, output_path)
    
    # Reset
    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("SpatiallyVarying", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def generateGroundTruthDataScene():
    """Generate ground truth lens simulation data for the scene."""
    Host.Print("\n=== Ground Truth Lens Simulation (Scene) ===")
    reset_all_variables()
    
    Host.SetVariable("RenderLensSimulationDoF", "true")
    Host.SetVariable("NumBounces", "16")
    Host.SetVariable("SamplesPerPixelPerFrame", "256")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    # Accumulate for convergence
    Host.Print(f"  Accumulating {GROUND_TRUTH_CONVERGE_FRAMES} frames for ground truth...")
    accumulate_frames(GROUND_TRUTH_CONVERGE_FRAMES, reset=True)
    
    # Save SDR tonemaped image
    output_sdr = os.path.join(BASE_OUTPUT_DIR, "scene", "ground_truth", "lens_simulation_ground_truth.png")
    save_image_png(RESOURCE_LENS_SIM_SDR, output_sdr)
    
    # Save HDR image
    output_hdr = os.path.join(BASE_OUTPUT_DIR, "scene", "ground_truth", "lens_simulation_ground_truth.exr")
    save_image_exr(RESOURCE_LENS_SIM_HDR, output_hdr)
    
    # Reset
    Host.SetVariable("RenderLensSimulationDoF", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

# Bokeh Config Profiling Functions

def letPinholeAccumulateBokehConfig():
    """Prepare bokeh config scene with pinhole rendering (no DoF)."""
    Host.Print("\n=== Pinhole Accumulation (Bokeh Config) ===")
    reset_all_variables()
    
    # Turn Bokeh Config on with NoDoF mode
    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("NumBounces", "2")
    Host.SetVariable("BokehConfigMode", "NoDoF")
    Host.SetVariable("ConfigOnlyDiagonal", "false")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    # Run for warmup frames
    warmup_frames = 50  # Smaller amount for bokeh config
    Host.Print(f"  Running {warmup_frames} warmup frames...")
    accumulate_frames(warmup_frames, reset=True)
    
    # Save rendered pinhole scene image as EXR
    output_path = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "images", "pinhole_no_dof.exr")
    save_image_exr(RESOURCE_BOKEH_CONFIG_HDR, output_path)
    
    # Keep bokeh config on but disable accumulation
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileGatherDoFSpatiallyConstantBokehConfig():
    """Profile Gather DoF with spatially constant settings for bokeh config."""
    Host.Print("\n=== Gather DoF Spatially Constant (Bokeh Config) ===")
    
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    
    # Run profiling
    profiling_data = run_profiling("GatherDoF_SpatiallyConstant_BokehConfig")
    output_dir = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "profiling")
    save_profiling_data(profiling_data, output_dir, "gather_dof_spatially_constant")
    
    # Accumulate to convergence and save rendered image
    Host.SetVariable("DoDOFAccumulation", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    Host.Print(f"  Accumulating {DOF_ACCUMULATION_FRAMES} frames...")
    accumulate_frames(DOF_ACCUMULATION_FRAMES, reset=True)
    
    output_path = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "images", "gather_dof_spatially_constant.png")
    save_image_png(RESOURCE_BOKEH_CONFIG_SDR, output_path)
    
    # Reset
    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileGatherDoFSpatiallyVaryingBokehConfig():
    """Profile Gather DoF with spatially varying settings for bokeh config."""
    Host.Print("\n=== Gather DoF Spatially Varying (Bokeh Config) ===")
    
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "true")
    Host.SetVariable("DoDOFAccumulation", "false")
    
    # Run profiling
    profiling_data = run_profiling("GatherDoF_SpatiallyVarying_BokehConfig")
    output_dir = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "profiling")
    save_profiling_data(profiling_data, output_dir, "gather_dof_spatially_varying")
    
    # Accumulate to convergence and save rendered image
    Host.SetVariable("DoDOFAccumulation", "true")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    Host.Print(f"  Accumulating {DOF_ACCUMULATION_FRAMES} frames...")
    accumulate_frames(DOF_ACCUMULATION_FRAMES, reset=True)
    
    output_path = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "images", "gather_dof_spatially_varying.png")
    save_image_png(RESOURCE_BOKEH_CONFIG_SDR, output_path)
    
    # Reset
    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("SpatiallyVarying", "false")
    Host.SetVariable("DoDOFAccumulation", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def generateGroundTruthDataBokehConfig():
    """Generate ground truth realistic lens data for bokeh config."""
    Host.Print("\n=== Ground Truth Realistic Lens (Bokeh Config) ===")
    reset_all_variables()
    
    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("NumBounces", "2")
    Host.SetVariable("BokehConfigMode", "RealisticLens")
    Host.SetVariable("ConfigOnlyDiagonal", "false")
    Host.SetVariable("Accumulate", "true")
    Host.SetVariable("Animate", "true")
    
    # Accumulate for convergence
    Host.Print(f"  Accumulating {GROUND_TRUTH_CONVERGE_FRAMES} frames for ground truth...")
    accumulate_frames(GROUND_TRUTH_CONVERGE_FRAMES, reset=True)
    
    # Save SDR tonemaped image
    output_sdr = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "ground_truth", "realistic_lens_ground_truth.png")
    save_image_png(RESOURCE_BOKEH_CONFIG_LENS_SDR, output_sdr)
    
    # Save HDR image
    output_hdr = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "ground_truth", "realistic_lens_ground_truth.exr")
    save_image_exr(RESOURCE_BOKEH_CONFIG_HDR, output_hdr)
    
    # Reset
    Host.SetVariable("RenderBokehConfig", "false")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def profileAndGenerateDOFBokehConfigWithVaryingNoiseLevels():
    """Profile and generate DoF images with varying tap counts (noise levels)."""
    Host.Print("\n=== Tap Count Comparison (Bokeh Config) ===")
    
    # Prepare pinhole base
    letPinholeAccumulateBokehConfig()
    
    Host.SetVariable("DoGatherDoF", "true")
    Host.SetVariable("SpatiallyVarying", "true")
    Host.SetVariable("DoDOFAccumulation", "false")
    
    output_dir = os.path.join(BASE_OUTPUT_DIR, "bokeh_config", "tap_count_comparison")
    all_profiling_results = []
    
    for tap_count in TAP_COUNT_VALUES:
        Host.Print(f"\n  Processing tap count: {tap_count}")
        Host.SetVariable("GatherDOFTapCount", str(tap_count))
        
        # Run profiling for this tap count
        profiling_data = run_profiling(f"GatherDoF_TapCount_{tap_count}")
        save_profiling_data(profiling_data, output_dir, f"tap_count_{tap_count:02d}")
        all_profiling_results.append(profiling_data)

        Host.RunTechnique()
        Host.WaitOnGPU()
        
        output_path = os.path.join(output_dir, f"gather_dof_tap_count_{tap_count:02d}.png")
        save_image_png(RESOURCE_BOKEH_CONFIG_SDR, output_path, f"(tap_count={tap_count})")
    
    # Save combined comparison summary
    save_tap_count_comparison_summary(all_profiling_results, output_dir)
    
    # Reset
    Host.SetVariable("DoGatherDoF", "false")
    Host.SetVariable("SpatiallyVarying", "false")


def save_tap_count_comparison_summary(profiling_results, output_dir):
    """Save a summary comparing all tap count profiling results."""
    summary_path = os.path.join(output_dir, "tap_count_comparison_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write("Tap Count Comparison Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Tap Count':<12} {'Total GPU (ms)':<15} {'Blur GPU (ms)':<15}\n")
        f.write("-" * 60 + "\n")
        
        for result in profiling_results:
            label = result["label"]
            tap_count = label.split("_")[-1]
            summary = result.get("summary", {})
            
            total_gpu = summary.get("Total", {}).get("gpu_avg_ms", 0)
            
            # Find blur pass (may vary by name)
            blur_gpu = 0
            for pass_name, stats in summary.items():
                if "Blur" in pass_name:
                    blur_gpu += stats.get("gpu_avg_ms", 0)
            
            f.write(f"{tap_count:<12} {total_gpu:<15.3f} {blur_gpu:<15.3f}\n")
    
    Host.Print(f"  Saved comparison summary: tap_count_comparison_summary.txt")

# Main Profiling Orchestration

def profileAll():
    """Run all profiling tasks."""
    Host.Print("\n" + "=" * 70)
    Host.Print("  BOKEH TECHNIQUE PROFILING - FULL SUITE")
    Host.Print("=" * 70)
    Host.Print(f"Output directory: {BASE_OUTPUT_DIR}")
    
    # Create output directories
    create_output_dirs()
    
    # Initialize - ensure everything is in a clean state
    reset_all_variables()
    Host.RunTechnique()
    Host.WaitOnGPU()
    
    # Scene Profiling
    Host.Print("\n" + "=" * 70)
    Host.Print("  SCENE PROFILING")
    Host.Print("=" * 70)
    
    letPinholeAccumulateScene()
    profileGatherDoFSpatiallyConstantScene()
    profileGatherDoFSpatiallyVaryingScene()
    generateGroundTruthDataScene()
    
    # Bokeh Config Profiling
    Host.Print("\n" + "=" * 70)
    Host.Print("  BOKEH CONFIG PROFILING")
    Host.Print("=" * 70)
    
    letPinholeAccumulateBokehConfig()
    profileGatherDoFSpatiallyConstantBokehConfig()
    profileGatherDoFSpatiallyVaryingBokehConfig()
    generateGroundTruthDataBokehConfig()
    
    # Special Comparisons - Tap Count Variation
    Host.Print("\n" + "=" * 70)
    Host.Print("  SPECIAL COMPARISONS - TAP COUNT VARIATION")
    Host.Print("=" * 70)
    
    profileAndGenerateDOFBokehConfigWithVaryingNoiseLevels()
    
    # Cleanup and Summary
    reset_all_variables()

    Host.Print("\n" + "=" * 70)
    Host.Print("  PROFILING COMPLETE")
    Host.Print("=" * 70)
    Host.Print(f"Results saved to: {BASE_OUTPUT_DIR}")
    Host.Print("=" * 70 + "\n")


def profileSceneOnly():
    """Run only scene profiling tasks."""
    Host.Print("\n=== Scene Profiling Only ===")
    create_output_dirs()
    reset_all_variables()
    Host.RunTechnique()
    Host.WaitOnGPU()
    
    letPinholeAccumulateScene()
    profileGatherDoFSpatiallyConstantScene()
    profileGatherDoFSpatiallyVaryingScene()
    generateGroundTruthDataScene()
    
    reset_all_variables()
    Host.Print(f"\nScene profiling complete. Results: {BASE_OUTPUT_DIR}")


def profileBokehConfigOnly():
    """Run only bokeh config profiling tasks."""
    Host.Print("\n=== Bokeh Config Profiling Only ===")
    create_output_dirs()
    reset_all_variables()
    Host.RunTechnique()
    Host.WaitOnGPU()
    
    letPinholeAccumulateBokehConfig()
    profileGatherDoFSpatiallyConstantBokehConfig()
    profileGatherDoFSpatiallyVaryingBokehConfig()
    generateGroundTruthDataBokehConfig()
    
    reset_all_variables()
    Host.Print(f"\nBokeh config profiling complete. Results: {BASE_OUTPUT_DIR}")


def profileTapCountComparisonOnly():
    """Run only tap count comparison."""
    Host.Print("\n=== Tap Count Comparison Only ===")
    create_output_dirs()
    reset_all_variables()
    Host.RunTechnique()
    Host.WaitOnGPU()
    
    profileAndGenerateDOFBokehConfigWithVaryingNoiseLevels()
    
    reset_all_variables()
    Host.Print(f"\nTap count comparison complete. Results: {BASE_OUTPUT_DIR}")


# Main Execution

# This is so the script can be run by itself directly in Gigi
if __name__ == "builtins":
    Host.Print("Starting Bokeh Profiling Script...")
    
    try:
        profileAll()
        Host.Log("Info", "Bokeh Profiling: All tests passed successfully")
    except Exception as e:
        Host.Log("Error", f"Bokeh Profiling: Error occurred - {str(e)}")
        raise

