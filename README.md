Aurora scale sizes

The main function is aurora_power.
To get the metres per pixel: if the field of view (FOV) is $\theta$ radians, and you have NxN pixels in the FOV:
$mpp \approx h\times 1000 \times tan(\frac{\theta}{N})$
where h is the emission height in km (e.g. 100 km).
Set mpp_base to something comparable to mpp, say rounded to the nearest 10 or 5.

The plotting function gives a quick overview of the output.

Tested on python3.9\n
Tested with following dependencies:\n
numpy 1.24.4\n
skimage 0.24.0\n
sklearn 1.5.0\n
scipy 1.11.3\n
matplotlib 3.7.0\n
