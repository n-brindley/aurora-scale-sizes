Aurora scale sizes:

A heuristic approach using a combination of existing image processing methods to get scale-dependent power in aurora. Has certain advantages over FFT and wavelet methods, but has no mathematical basis. This method has adjustable parameters to increase/decrease robustness to noise at the expense of decreased/increased detection of subtle structures. 

The main function is aurora_power. It works best on discrete and bright emissions.

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

