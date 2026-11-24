Fiber end-face D4σ calculation with a Gaussian aperture correction

Reference
L. R. Hofer, R. V. Dragone, A. D. MacGregor, Scale factor correction for
Gaussian beam truncation in second moment beam radius measurements,
Optical Engineering 56(4), 043110 (2017).
https://doi.org/10.1117/1.OE.56.4.043110

Overview
This script evaluates the radial D4σ beam width under a finite field of view
using the circular-aperture Gaussian correction in Eq. (23) of Hofer et al.
For comparison, it prints the 1/e² intensity-contour diameter, 
the 1/e²-limited D4σ diameter (illustrating the effect of excluding the outer wings),
the direct inscribed-aperture D4σ diameter, and the Hofer-corrected final diameter. 
Signed background-subtracted residuals are retained for all D4σ integrals. 
The package includes synthetic demonstration images; 
raw experimental laser images are withheld due to institutional restrictions.
The experimental value of 94.88 µm is reproduced separately from the measured
radial profiles and the inscribed circular aperture with the Hofer correction.
The synthetic examples illustrate the effects of truncation and of the Gaussian aperture correction.

Requirements
Python 3.10 or later, NumPy, and Pillow.
Install with: python3 -m pip install numpy Pillow

Run the bundled example images
  python3 measure_endface_d4sigma.py
  python3 measure_endface_d4sigma.py --csv synthetic_results.csv

The script reads all three PNG files in synthetic_demo_images/ next to the
script and uses their example pixel scale of 0.057 ± 0.001 µm/px.
The files are 2560x2560 16-bit grayscale surrogate frames
with a flat-top + halo radial profile (order-6 core with wc = 46.1 µm and
order-3 halo of 25% peak amplitude with wh = 63.1 µm, scaled uniformly by ±0.5 µm in wc across the three frames). 

Expected synthetic example results
- Mean 1/e² diameter: 98.132 µm (1721.61 px)
- Mean 1/e²-limited D4σ: 84.194 µm (1477.08 px)
- Mean aperture-integrated D4σ: 92.264 µm (1618.66 px)
- Final mean Hofer-corrected D4σ: 94.859 µm (1664.19 px)
- Repeatability u_A: 0.581 µm
- Calibration u_cal: 0.961 µm
- Partial combined u_c: 1.123 µm; expanded U (k=2): 2.246 µm

Run on another image directory
  python3 measure_endface_d4sigma.py --image-dir "/path/to/images" --scale-um-per-pixel <scale> --csv results.csv

Notes
Eq. (23) is derived for a Gaussian beam and agrees within 0.3% with the
untruncated reference for the flat-top + halo demo profile at R_ap / w ≈ 1.48.
The reported uncertainty covers repeatability and supplied scale calibration only.
