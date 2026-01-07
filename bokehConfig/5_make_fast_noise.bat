pushd "C:\Users\ckobalt\Documents\MasterThesis\fastnoise"
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 bokeh_fl45.0_as6_samples1000000_od400_lidx0of15 -split -init "C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\4_blue_noise_bin\bokeh_fl45.0_as6_samples1000000_od400_lidx0of15.bin" -progress 1 -output exr
for %%f in ("bokeh_fl45.0_as6_samples1000000_od400_lidx0of15*") do move /Y "%%f" "C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\5_fast_noise"
popd