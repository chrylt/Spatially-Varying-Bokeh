- This code is provided without any guarantees; please review the code first.

- Please feel free to use it as suits your needs.

- Recognition of the author is highly appreciated.

- I provided a photo of myself for testing.

- Please check my website, abdallagafar.com, for updates.

- I am a researcher but not a professional programmer, so I am relying on
individual files to demonstrate the core algorithms. I may consider building a proper make utility in the future.

- Here is an example compilation line:

    nvcc -Xcompiler -O3,-march=native,-msse4.1 -o spectrum spectrum.cu -lcairo


and here are example run lines:

    ./gbn-adaptive taksim-circle.pgm 10000 1000 test.{pdf,png}

    ./gbn-bounded 1024 1000 test.txt && ./spectrum test.txt -o test.png -r test-rp.tex && pdflatex test-rp.tex

    ./gbn-toroidal 1024 1000 test.txt && ./spectrum test.txt -o test.png -r test-rp.tex && pdflatex test-rp.tex

    ./gbn-reconstruct -g 0.43 -w 512 -a 187 -i 15 test.txt test.png

