This code and the text below is from Abdalla Gafar abdalla_gafar@hotmail.com

================================================================================

It is worth noting that the original version of adaptive GBN (algorithm 2 in the 2022 paper) is suboptimal. Specifically, it does not handle the boundaries well, besides being a bit complex. It also does not allow an objective choice of sigma to match the unform case.

Since them, the model has undergone two further developments. One is for the 2023 SIGGraph poster (code not published), and the last one is published in a recent short paper https://doi.org/10.2312/cgvc.20241226

The model in this version is quite simpler and far more stable, along with a straighforward reconstruction model. Further, it admits serial (CPU) implementation. For best results I therefore strongly recommend using this variant, possibly citing the mentioned paper instead/in addition. Please find attached my current gpu and cpu implementations.

The parameter space is admittedly large, as I am trying to find the most optimal parameters. Actually I am right now working on a (hopefully) improved version.
The default settings should word well.
You may want to try the following options for sigma:
1: this is my default. It tends to align the points in a quad mesh, with a good balance between fill and edges.

sqrt(sqrt(3)/2)): this tends to produce a triangular grid instead, with more emphasis on fill.

sqrt(Pi/4): the most greedy setting for sigma; no edges. Smaller sigma tends to leave gaps in the coverage.
sqrt(2/sqrt(3)): The corresponding upper limit, with more emphasis on edges. Larger sigma tends to produce clusters.
These settings are basically empirical but with some plausible justifications that we may discuss if you are interested.

Please let me know if you have questions regarding this code.