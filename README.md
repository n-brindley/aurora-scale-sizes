Aurora scale sizes

The main function is aurora_power.

To get the metres per pixel: if the field of view (FOV) is $\theta$ radians, and you have NxN pixels in the FOV:

$mpp \approx h\times 1000 \times tan(\frac{\theta}{N})$,

where h is the emission height in km (e.g. 100 km).

Set mpp_base to something comparable to mpp, say rounded to the nearest 10 or 5.

The plotting function gives a quick overview of the output.

Tested on python3.9

Tested with following dependencies:

numpy 1.24.4

skimage 0.24.0

sklearn 1.5.0

scipy 1.11.3

matplotlib 3.7.0

