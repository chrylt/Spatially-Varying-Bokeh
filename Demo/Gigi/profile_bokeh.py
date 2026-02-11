import Host
import os
import json
from datetime import datetime

# Profiling settings
num_warmup_frames = 5
num_profiling_frames = 100

# Don't save gguser files during this script execution
Host.DisableGGUserSave(True)

# Log file handle (initialized in main)
log_file = None

def log(msg):
    """Write message to log file and Gigi console."""
    Host.Print(str(msg))
    if log_file:
        log_file.write(str(msg) + "\n")
        log_file.flush()

def init():
    """Initialize the technique with default settings for profiling."""
    Host.SetVariable("SmallLightBrightness", "100")
    Host.SetVariable("ToggleChromaticAberration", "false")
    Host.SetVariable("RenderBokehConfig", "true")
    Host.SetVariable("BokehConfigMode", "RealisticLens")
    
    # Turn off other renderers for accurate profiling of target technique
    Host.SetVariable("RenderPinhole", "false")
    Host.SetVariable("RenderThinLensDoF", "false")
    Host.SetVariable("RenderLensSimulationDoF", "false")
    Host.SetVariable("NumBounces", "2")
    
    # Default rendering settings
    Host.SetVariable("FilmDistanceToLens", "38.029")
    Host.SetVariable("HeliosApertureStop", "6")
    Host.SetVariable("Accumulate", "false")
    Host.SetVariable("Animate", "false")

def run_profiling():
    """Run profiling and collect timing data."""
    log(f"Starting profiling: {num_warmup_frames} warmup frames, {num_profiling_frames} profiling frames")
    
    # Turn on profiling mode
    Host.SetProfilingMode(True)
    Host.ForceEnableProfiling(True)
    
    # Warmup: run a few frames to ensure everything is initialized
    Host.RunTechnique(num_warmup_frames)
    Host.WaitOnGPU()
    log(f"Warmup complete ({num_warmup_frames} frames)")
    
    # Collect profiling data
    profiling_results = []
    
    for i in range(num_profiling_frames):
        Host.RunTechnique()
        data = Host.GetProfilingData()
        
        # Store frame data
        frame_data = {
            "frame": i,
            "timings": {}
        }
        
        # GetProfilingData returns a dict with pass names as keys
        # Each value is typically [cpu_time, gpu_time] or similar
        for key, value in data.items():
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                frame_data["timings"][key] = {
                    "cpu_ms": value[0],
                    "gpu_ms": value[1]
                }
            else:
                frame_data["timings"][key] = value
        
        profiling_results.append(frame_data)
        
        if (i + 1) % 20 == 0:
            log(f"  Profiled {i + 1}/{num_profiling_frames} frames")
    
    # Turn off profiling mode
    Host.SetProfilingMode(False)
    
    return profiling_results

def compute_statistics(profiling_results):
    """Compute statistics from profiling results."""
    if not profiling_results:
        return {}
    
    # Gather all timing keys
    all_keys = set()
    for frame in profiling_results:
        all_keys.update(frame["timings"].keys())
    
    stats = {}
    for key in all_keys:
        gpu_times = []
        cpu_times = []
        
        for frame in profiling_results:
            timing = frame["timings"].get(key)
            if timing and isinstance(timing, dict):
                if "gpu_ms" in timing:
                    gpu_times.append(timing["gpu_ms"])
                if "cpu_ms" in timing:
                    cpu_times.append(timing["cpu_ms"])
        
        if gpu_times:
            gpu_times_sorted = sorted(gpu_times)
            stats[key] = {
                "gpu_avg_ms": sum(gpu_times) / len(gpu_times),
                "gpu_min_ms": min(gpu_times),
                "gpu_max_ms": max(gpu_times),
                "gpu_median_ms": gpu_times_sorted[len(gpu_times_sorted) // 2],
                "samples": len(gpu_times)
            }
            if cpu_times:
                stats[key]["cpu_avg_ms"] = sum(cpu_times) / len(cpu_times)
    
    return stats

def save_results(profiling_results, stats):
    """Save profiling results to files."""
    # Create output directory
    out_dir = os.path.join(Host.GetScriptPath(), "profiling")
    os.makedirs(out_dir, exist_ok=True)
    
    # Generate timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results as JSON
    detail_path = os.path.join(out_dir, f"profile_detail_{timestamp}.json")
    with open(detail_path, "w") as f:
        json.dump({
            "settings": {
                "warmup_frames": num_warmup_frames,
                "profiling_frames": num_profiling_frames
            },
            "frames": profiling_results,
            "statistics": stats
        }, f, indent=2)
    log(f"Saved detailed results: {detail_path}")
    
    # Save summary as readable text
    summary_path = os.path.join(out_dir, f"profile_summary_{timestamp}.txt")
    with open(summary_path, "w") as f:
        f.write("Bokeh Technique Profiling Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Warmup frames: {num_warmup_frames}\n")
        f.write(f"Profiling frames: {num_profiling_frames}\n\n")
        f.write("GPU Timings (ms):\n")
        f.write("-" * 50 + "\n")
        f.write(f"{'Pass':<30} {'Avg':>8} {'Min':>8} {'Max':>8}\n")
        f.write("-" * 50 + "\n")
        
        # Sort by average GPU time descending
        sorted_stats = sorted(stats.items(), key=lambda x: x[1].get("gpu_avg_ms", 0), reverse=True)
        for key, s in sorted_stats:
            avg = s.get("gpu_avg_ms", 0)
            min_val = s.get("gpu_min_ms", 0)
            max_val = s.get("gpu_max_ms", 0)
            f.write(f"{key:<30} {avg:>8.3f} {min_val:>8.3f} {max_val:>8.3f}\n")
    
    log(f"Saved summary: {summary_path}")
    
    return detail_path, summary_path

def main():
    """Main profiling entry point."""
    global log_file
    
    # Create output directory and log file
    out_dir = os.path.join(Host.GetScriptPath(), "profiling")
    os.makedirs(out_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(out_dir, f"profile_log_{timestamp}.txt")
    log_file = open(log_path, "w")
    
    log("Initializing technique for profiling...")
    init()
    
    log("\nRunning profiling...")
    results = run_profiling()
    
    log("\nComputing statistics...")
    stats = compute_statistics(results)
    
    log("\nSaving results...")
    save_results(results, stats)
    
    # Log quick summary
    log("\n" + "=" * 50)
    log("Quick Summary (Top 5 by GPU time):")
    log("=" * 50)
    sorted_stats = sorted(stats.items(), key=lambda x: x[1].get("gpu_avg_ms", 0), reverse=True)[:5]
    for key, s in sorted_stats:
        log(f"  {key}: {s.get('gpu_avg_ms', 0):.3f} ms avg")
    
    log("\nProfiling complete!")
    log(f"Log saved to: {log_path}")
    
    log_file.close()
    log_file = None

# Run when executed as script in Gigi
if __name__ == "builtins":
    main()
