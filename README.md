# Implementation Details

## Pipeline

### Rendering and processing reference bokeh shapes

First, the reference bokeh shapes from the lens simulation need to be generated along the diagonal by running `bokehConfig/1_gigi_bokeh_config_gen.py` directly from Gigi. This places the one-by-one rendered shapes as .exr images in the folder `bokehConfig/1_rawRenderings`.

In `bokehConfig/1_gigi_bokeh_config_gen.py`, we set the focus distance to 45 units and the light spheres of radius 0.3 at a distance of 250 units. We set the f-stop to its widest setting and generate 15 shapes, each in a different image along the half-diagonal, with 8192 SPP until an accumulated total of 134217728 SPP is reached. We save these images as .exr files, which are shown below.

<p align="center">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light1of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light2of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light3of15.png" width="32%">
</p>
<p align="center">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light4of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light5of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light6of15.png" width="32%">
</p>
<p align="center">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light7of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light8of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light9of15.png" width="32%">
</p>
<p align="center">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light10of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light11of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light12of15.png" width="32%">
</p>
<p align="center">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light13of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light14of15.png" width="32%">
  <img src="figures/bokeh_focus45.0_aperture6_samples167772216_distance250_light15of15.png" width="32%">
</p>

*Figure: Raw Filter Kernel Density Renderings — The raw renderings of the 15 bokeh shapes along the half-diagonal. The last two shapes intersect with the image edge and corner and are thus discarded. Here we can perceive the image location of the rendered shapes, for a more detailed look on the individual shapes, refer to the cropped shapes below.*

Next, we run the jupyter notebook `bokehConfig/2_process_raw_bokeh.ipynb`. This crops all bokeh shapes to the same size, which is defined by the largest bokeh size, the center bokeh. The center of each cropped bokeh image is calculated as an intensity-weighted center of the shape. We save these cropped bokeh shapes (see below) as .pgm in the folder `2_cropped`.

<p align="center">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light1of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light2of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light3of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light4of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light5of15.png" width="19%">
</p>
<p align="center">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light6of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light7of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light8of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light9of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light10of15.png" width="19%">
</p>
<p align="center">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light11of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light12of15.png" width="19%">
  <img src="figures/cropped/bokeh_focus45.0_aperture6_samples167772216_distance250_light13of15.png" width="19%">
</p>

*Figure: Rendered Filter Kernel Densities — The bokeh shapes are cropped to the same shape defined by the center-most shape. The shapes that touched the edge of the raw rendering were removed as they could not be used meaningfully. Instead, we use the last shape for those corner areas. Odd indices will be used as key shapes for the distortion, and even indices will be used as mid-points in the gradient descent.*

In this notebook, we also calculate the intensity sum for each shape and fit a polynomial to the values as a function of their distance from the non-cropped image center. We print the resulting coefficients and use them in the code for the intensity correction.

#### Generating noise

As a last step in `bokehConfig/2_process_raw_bokeh.ipynb`, we generate `3_generate_noise.bat`, which generates FAST noise for the center bokeh shape in 3 steps, similar to Wolfe et al. [[1]](#references). First, blue noise is generated by running the utility `agbn.exe`, then it is packed to binary format, and finally, FAST noise is generated via a clone from the GitHub of Donnelly et al. [[2]](#references). For details and parameters, see the batch script below.

<details>
<summary>Batch script: generate blue-noise, pack to binary, and generate FAST noise</summary>

```batch
echo Step 1: Generating blue noise...
for /l %%x in (0, 1, 31) do ..\Utils\agbn\agbn.exe -F 0 path\Spatially-Varying-Bokeh\bokehConfig\2_cropped\bokeh_focus45.0_aperture6_samples167772216_distance250_light1of15.pgm 16384 1000 path\Spatially-Varying-Bokeh\bokehConfig\3_blue_noise\base_bokeh_%%x.txt path\Spatially-Varying-Bokeh\bokehConfig\3_blue_noise\base_bokeh_%%x.png

echo Step 2: Packing blue noise to bin...
..\Utils\Gbn2Bin\x64\Release\Gbn2Bin.exe path\Spatially-Varying-Bokeh\bokehConfig\3_blue_noise\base_bokeh_ path\Spatially-Varying-Bokeh\bokehConfig\4_blue_noise_bin\base_bokeh

echo Step 3: Running FastNoise...
pushd "path\fastnoise"
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 base_bokeh -split -init "path\Spatially-Varying-Bokeh\bokehConfig\4_blue_noise_bin\base_bokeh.bin" -progress 1
for %%f in ("base_bokeh*") do move /Y "%%f" "path\Spatially-Varying-Bokeh\bokehConfig\5_fast_noise"
popd

echo Done!
pause
```

</details>

After running `bokehConfig/2_process_raw_bokeh.ipynb`, we need to actually generate the noise by running `3_generate_noise.bat`. This takes a while and populates folders `3_blue_noise`, `4_blue_noise_bin` and `5_fast_noise`. After the FAST noise is generated, we copy it to the Gigi project's assets folder, namely `Assets/NoiseTextures/bokeh/`, for use in the code.

#### Generating distortion maps

We generate the distortion flow fields by running the Jupyter notebook `distortion_vectors.ipynb`.
It accesses the cropped bokeh .pgm files in `2_cropped` and generates the distortion maps in approximately 40 seconds.
Finally, the generated distortion maps are saved in the `distortion_maps` folder and must also be copied to the Gigi project's asset folder `Assets/DistortionMaps`.

## Gigi Project

For the Gigi project, we modified the implementation of Wolfe et al. [[1]](#references) for our purposes. Our Gigi graph can be seen below.

We structured our project to enable direct runtime comparison of different DoF techniques by using separate buffers for their outputs. It is possible to run all at the same time by setting their specific rendering toggle in the Gigi Viewer to *true*, but we don't recommend it due to performance issues.

It is possible to change a variety of parameters in the Gigi Viewer to see their effects in real-time, including but not limited to aperture stop and focus distance in the lens simulation, grid properties of the BokehConfig scene, to render a grid or diagonal of bokeh shapes, and their density or parameters for the different GatherDoFs.

<p align="center">
  <img src="figures/gigi_project.PNG" width="100%">
</p>

*Figure: Gigi Project Graph — The graph of our Gigi project shows how different components interact with each other.*

### Overview of relevant source files

1. **`CameraLensData.hlsli`** — This file contains the lens data for the simulation as well as the precise values from which the data was derived.

2. **`LensSimulation.hlsli`** — Contains the physically based lens simulation code, where rays are traced from the film through the lens into the scene. Chromatic aberration is also implemented here.

3. **`RT_exterior.hlsl`** — Adapted from Wolfe et al. [[1]](#references) to incorporate the lens simulation.

4. **`SpatiallyVaryingBokeh.hlsli`** — Contains all the code that distorts samples according to our distortion maps and is integrated into the DoF by modifications in `BlurFarCS.hlsl` and `NearBlurCS.hlsl`.

## References

1. Wolfe et al., "ISFASS" (2025)
2. Donnelly et al., "FAST" (2023)

