# Copilot Instructions

## Project Snapshot
- Core shaders live in `Demo/Gigi`, with `RT_exterior.hlsl` orchestrating the ray-generation pass for the Gigi scene.
- Shader parameters are provided through Frostbite/Helios tokens (`/*$(Variable:...)*/`, `/*$(Image2D:...)*/`); never rename or strip these placeholders.
- Supporting includes: `LensSimulation.hlsli` for physical lens tracing, `DrawBokehConfig.hlsli` for synthetic light-field probes, `CameraLensData.hlsli` for lens element definitions, and `DrawDebugLens.hlsli` for UI overlays via the `s2h` helpers.
- No CPU-side build scripts are present here; shaders are compiled and hot-reloaded through the host viewer that consumes this demo content.

## Render Paths
- `RT_exterior.hlsl` now emits separate UAVs: `PinholeOut`, `ThinlensOut`, `LensSimulationOut`, `BokehConfigOut`, and `DebugLensOut`.
- Feature toggles (`RenderPinhole`, `RenderThinLensDoF`, `RenderLensSimulationDoF`, `RenderBokehConfig`) decide which paths run per pixel.
- Thin-lens depth of field flows through `ApplyDOFLensSimulation`; it only perturbs rays when `DOFMode::PathTraced` is selected.
- Realistic lens rendering calls `TraceRealisticMonochrome`/`TraceRealisticChromatic`, which now accept a `bokehView` flag to switch between scene shading and visual-field previews.
- `ToggleChromaticAberration` selects between monochrome and RGB-split sampling for both the lens-simulation output and the bokeh-config preview.

## Sampling & Data Flow
- Each ray sample seeds RNGs via `HashInit` and then derives per-technique streams with `wang_hash`; keep that pattern when adding stochastic work.
- Incremental averaging uses `lerp(accum, sample, 1/(rayIndex+1))`, followed by frame-to-frame blending controlled by `FrameIndex`, `Accumulate`, and `Animate`.
- `LinearDepth` is sourced exclusively from the pinhole path (`pinholeDebug.HitT`) to preserve compatibility with downstream compositors.
- Visual-field previews rely on `ShadeVisualFieldSample`, which calls `DrawBokehConfig.hlsli::VisualFieldLightContributions`; extend that function when adding new light-layout experiments.
- Lens metadata comes from the `lens_elements` array in `CameraLensData.hlsli`; update both curvature/thickness data and related radius tables when authoring new optics.

## Debugging & Validation
- `Struct_PixelDebugStruct` is populated per path; the ray-gen pass prioritizes realistic lens data, then thin-lens, then pinhole when writing to `PixelDebug`.
- `DrawDebugLens.hlsli` composites a 2D lens cross-section into `DebugLensOut`; keep `DebugInfo` fields in sync if you extend diagnostic rendering.
- The retina of small debug lights is handled in `SmallLightContributions`; reuse it for quick visual probes before touching the full material path tracer.
- To validate changes, launch the Gigi demo in the viewer, toggle the render flags individually, and inspect each UAV via the tooling (RenderDoc, in-engine debug panels, or the Helios UI).
- Maintain ASCII-only edits unless the existing file already relies on Unicode characters.
