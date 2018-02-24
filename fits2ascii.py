"""
   This routine converts a FITS table file (NOT IMAGES)
   to ASCII files, saving each column into a new ASCII file

                          Marcus - 24/02/2018 
"""

import numpy as np
import astropy.io.fits as fits
import sys

def fits2ascii(infile, outfile):

    print 
    print '>> Opening FITS file <<'
    print infile

    hdulist = fits.open(infile)

    # Load data

    data = hdulist[1].data

    # Number of rows 

    nrow = np.shape(data)[0]
    print 'nrow=', nrow

    # Columns names

    col_name = hdulist[1].columns.names
    hdulist.close() # close file

    print 
    print '>> Output <<'
    print outfile

    f = open(outfile, 'w')

    # header

    string = ' '.join(col_name[j] for j in range(len(col_name))) + ' \n'
    f.writelines(string)

    for i in range(nrow): # row-by-row

        if i % 1000 == 0: print i, nrow
        
        string = ' '.join(str(data[col_name[j]][i]) for j in range(len(col_name))) + ' \n'
        f.writelines(string)
    f.close()

    return

if __name__ == "__main__":

    fits2ascii(sys.argv[1], sys.argv[2])
