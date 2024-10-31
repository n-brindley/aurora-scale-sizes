Aurora scale sizes:

A heuristic approach using a combination of gaussian mixture models and existing image processing techniques to get scale-dependent power in aurora. This method has certain advantages over FFT and wavelet methods, but has no mathematical basis. It has adjustable parameters to increase/decrease robustness against noise at the expense of decreased/increased detection of subtle structures.

Advantages:

–No need to assume underlying signal (cf. needing to choose mother wavelet in wavelet transform).

–Adjustable robustness against noise.

–Avoids spurious power at large scales when the structures are actually narrow (cf. FFT methods, where, for example, the FFT of a gaussian pulse gives another gaussian in wavenumber space, where the width in the wavenumber domain is inversely related to the width in the spatial domain; the highest power is still at large scales, even though the structure is narrow).

Disadvantages:

– Computation time.

– There is a slight bias towards finding small scale structures when the range of brightnesses in the image is particularly large, as the more gaussians are fitted in the GMM, the more likely it is that small structures will be identified which are not real structures, but are due to non-uniform edges of larger structures. The de-noising controlled by the optional parameters mitigates this effect, but you should check the output graphically to see whether it is significant.



Usage:

– We recommend sigma clipping images to remove stars prior to using the algorithm, but the de-noising has a good chance of removing them.

–The main function is aurora_power. It works best on discrete and bright emissions, but results will be reasonable in other cases. Images are meant to be in Rayleighs; behaviour has not been tested on raw counts.

– To get the metres per pixel: if the field of view (FOV) is $\theta$ radians, and you have NxN pixels in the FOV:

$mpp \approx h\times 1000 \times tan(\frac{\theta}{N})$,

where h is the emission height in km (e.g. 100 km).

– Set mpp_base to something comparable to mpp, say rounded down to the nearest 10 or 5.

– The plotting function gives a quick overview of the output.




Tested on python3.9

Tested with following dependencies:

– numpy 1.24.4

– skimage 0.24.0

– sklearn 1.5.0

– scipy 1.11.3

– matplotlib 3.7.0

