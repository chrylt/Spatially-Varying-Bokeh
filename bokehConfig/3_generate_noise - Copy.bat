echo Step 2: Packing blue noise to bin...
..\Utils\Gbn2Bin\x64\Release\Gbn2Bin.exe C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\3_blue_noise\base_bokeh_ C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\4_blue_noise_bin\base_bokeh

echo Step 3: Running FastNoise...
pushd "C:\Users\ckobalt\Documents\MasterThesis\fastnoise"
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 base_bokeh -split -init "C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\4_blue_noise_bin\base_bokeh.bin" -progress 1
for %%f in ("base_bokeh*") do move /Y "%%f" "C:\Users\ckobalt\Documents\MasterThesis\Spatially-Varying-Bokeh\bokehConfig\5_fast_noise"
popd

echo Done!
pause