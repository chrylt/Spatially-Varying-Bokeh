for /l %%x in (0, 1, 31) do .\agbn -o UniformStar.pgm 16384 1000 UniformStar%%x.txt UniformStar%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o Lens.pgm 16384 1000 Lens%%x.txt Lens%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o UniformCircle.pgm 16384 1000 UniformCircle%%x.txt UniformCircle%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o hexagon.pgm 16384 1000 hexagon%%x.txt hexagon%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o Lens_kernel_compositingpro.006.pgm 16384 1000 Lens_kernel_compositingpro.006.%%x.txt Lens_kernel_compositingpro.006.%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o NonUniformStar.pgm 16384 1000 NonUniformStar.%%x.txt NonUniformStar.%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o NonUniformStar2.pgm 16384 1000 NonUniformStar2.%%x.txt NonUniformStar2.%%x.png
for /l %%x in (0, 1, 31) do .\agbn -o UniformSquare.pgm 16384 1000 UniformSquare.%%x.txt UniformSquare.%%x.png

.\agbn -o Lens.pgm 8192 1000 Lens_kernel_compositingpro.204_8192.png
.\agbn -o Lens.pgm 4096 1000 Lens_kernel_compositingpro.204_4096.png
.\agbn -o Lens.pgm 2048 1000 Lens_kernel_compositingpro.204_2048.png
.\agbn -o Lens.pgm 1024 1000 Lens_kernel_compositingpro.204_1024.png
.\agbn -o Lens.pgm 256 1000 Lens_kernel_compositingpro.204_256.png