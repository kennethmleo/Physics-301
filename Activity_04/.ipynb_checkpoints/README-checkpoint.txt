Munsell colors matt (AOTF measured)

The reflectance spectra of 1250 matt Munsell color chips. Each spectrum consists of
61 elements that are raw data got from the output of the 12 bit A/D-converter of the
AOTF color measuring equipment. All the element values should be between 0 and
4096. However there can exist some values which are larger than 4096. Those element
values should be corrected to the value 4096. Furthermore there exists some noise at
the blue end of spectra. This was caused by the low intensity of the illuminator at
wavelengths below 420 nm. The illuminator was a standard halogen lamp and 0/0
geometry was used when the samples were measured.

Measuring equipment: Acousto Optic Tunable Filter (AOTF).

Wavelength interval: 400 nm - 700 nm.

Wavelength resolution: 5 nm.

File formats: The available formats are ASCII and MATLAB(tm).

The file munsell400_700_5.tar.gz consists of 40 ASCII files that are named after the
hue pages used in the Munsell Book of Color - Matte Finish Collection. So each file
includes the color spectra measured from the chips of the corresponding page. Each
spectrum consists of two lines. The first line is the label of the spectrum following
relatively closely the notation used in the Munsell Book of Color. For example the
first spectrum in file 10b is labeled as V90C01.NM5. Hence V90 means the value 9
(9/ in correct Munsell notation) and C01 means the chroma 1 (/1 in correct Munsell
notation). Suffix '.NM5' is the same for all spectra meaning the wavelength interval
between each component of the spectrum. The spectrum itself is stored on the second
line. It is measured from 400 to 700 nm at 5 nm intervals so each spectrum can be
considered as 61 dimensional vector at which the first component corresponds the
reflectance intensity at the wavelength 400nm, the second at 405 nm and so on the last
61th component corresponding the reflectance intensity at the wavelength 700nm.
Note that there is one empty line after each spectrum.

The file munsell400_700_5.mat.gz is a MATLAB(tm) file which includes 3 matrices
named:

   munsell	61x1250 matrix where each column is one 61 component spectrum.

   S		1250x15 matrix where 1st row is the label of 1st spectrum in
		munsell matrix, 2nd row is the label of 2nd spectrum and so on.

   C		16x1250 matrix where 1st column consists of the color coordinates
		of the 1st spectrum in munsell matrix, the 2nd column consists of
		the color coordinates of the 2nd spectrum and so on. The color 
		coordinates are calculated by using D65 light source and they are
		stored in the column vectors in following order:

        	C(1)    x
        	C(2)    y
        	C(3)    z
        	C(4)    X
        	C(5)    Y
        	C(6)    Z
        	C(7)    R
        	C(8)    G
        	C(9)    B
        	C(10)   L*
        	C(11)   a*
        	C(12)   b*
        	C(13)   u
        	C(14)   v
        	C(15)   u*
        	C(16)   v*

Measurer: University of Kuopio, Finland.

Further information: haanpalo@lut.fi

References: 

   Munsell Book of Color - Matte Finish Collection (Munsell Color, Baltimore,
   Md., 1976).

   Parkkinen, J. P. S., Hallikainen, J. and Jaaskelainen, T.: ``Characteristic spectra
   of Munsell colors,'' Journal of the Optical Society of America, Vol. 6, No. 2,
   February 1989, pp. 318-322.

   Jaaskelainen, T., Parkkinen, J. and Toyooka, S.: ``Vector-subspace model for
   color representation,'' Journal of the Optical Society of America, Vol. 7, No.
   4, April 1990, pp. 725-730.
 
   Parkkinen, J., Jaaskelainen, T. and Kuittinen, M.: ``Spectral representation of
   color images,'' IEEE 9th International Conference on Pattern Recognition,
   Rome, Italy, 14-17, 1988, Vol. 2, pp. 933-935.
 
   Parkkinen, J., Oja, E. and Jaaskelainen T.: ``Color analysis by learning
   subspaces and optical processing,'' IEEE International Conference on Neural
   Networks, San Diego, USA, July 24-27, 1988, Vol. 2, pp. 421-427.
 
   Hallikainen, J., Parkkinen, J. P. S. and Jaaskelainen, T.: ``Color image
   processing with AOTF,'' Proceedings of the 6th Scandinavian Conference on
   Image Analysis, Oulu, Finland, June 19-22, 1989. 
