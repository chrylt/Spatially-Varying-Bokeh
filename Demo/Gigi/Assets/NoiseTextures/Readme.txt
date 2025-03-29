How the noise textures were made...

======================== 1) Make blue noise point sets ========================

To make the blue noise point sets that go into the FAST textures, I ran the make file Utils/agbn/_MakeSamples.bat.

This uses greyscale .pgm files (you can open these files in gimp) to describe point densities for blue noise point sets.

It makes 128*128=16,384 points 32 times for each .pgm file.

This makes enough points to initialize a 128x128x32 texture, stored in text files.

The FAST optimizer only swaps points within the same Z slice, which is why we want 32 sets of good blue noise point sets,
instead of just a massive singular point set with 128x128x32 = 524,288 blue noise points in it.

======================== 2) Make bin files to init FAST noise initialization ========================

Utils/Gbn2Bin/Gbn2Bin.sln is then used to convert each 32 sets of text files into a binary packed .bin file.

Each .bin file contains 524,288 float4s, or 2,097,152 floats, written in binary, for a total of 8MB per file.

The z component is 0 and the w component is 1, as we are only working in 2D and only have data for x and y. These are the blue and alpha channels in the final image, respectively.

The points are shuffled before being written out, to ensure there is no correlation between points to start out with.

These files are then able to be used by FAST as initialization states before optimization.

======================== 3) Make noise textures ========================

I ran FastNoise.exe with these parameters, which take our .bin files as input, and also spit out both the initial and final textures.

Commands:
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 NonUniformStar -split -init NonUniformStar.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 NonUniformStar2 -split -init NonUniformStar2.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 Lens_kernel_compositingpro.006 -split -init Lens_kernel_compositingpro.006.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 Lens_kernel_compositingpro.204 -split -init Lens_kernel_compositingpro.204.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 UniformCircle -split -init UniformCircle.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 UniformHexagon -split -init UniformHexagon.bin -progress 1
FastNoise.exe Vector2 Uniform gauss 1.0 exponential 0.1 0.1 separate 0.5 128 128 32 UniformStar -split -init UniformStar.bin -progress 1


FastNoise.exe Vector2 Uniform box 3 exponential 0.1 0.1 separate 0.5 128 128 32 UniformHexagonBox3x3 -split -init UniformHexagon.bin -progress 1
FastNoise.exe Vector2 Uniform box 3 exponential 0.1 0.1 separate 0.5 128 128 32 UniformStarBox3x3 -split -init UniformStar.bin -progress 1
FastNoise.exe Vector2 Uniform box 3 exponential 0.1 0.1 separate 0.5 128 128 32 Lens_kernel_compositingpro.204Box3x3 -split -init Lens_kernel_compositingpro.204.bin -progress 1
