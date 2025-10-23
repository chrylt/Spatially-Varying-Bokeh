for /l %%x in (0, 1, 31) do ..\Utils\agbn\agbn.exe -o bokehInv.pgm 16384 1000 bokehInv%%x.txt bokehInv%%x.png

..\Utils\agbn\agbn.exe -o bokehInv.pgm 8192 1000 bokehInv_kernel_compositingpro.204_8192.png
..\Utils\agbn\agbn.exe -o bokehInv.pgm 4096 1000 bokehInv_kernel_compositingpro.204_4096.png
..\Utils\agbn\agbn.exe -o bokehInv.pgm 2048 1000 bokehInv_kernel_compositingpro.204_2048.png
..\Utils\agbn\agbn.exe -o bokehInv.pgm 1024 1000 bokehInv_kernel_compositingpro.204_1024.png
..\Utils\agbn\agbn.exe -o bokehInv.pgm 256 1000 bokehInv_kernel_compositingpro.204_256.png