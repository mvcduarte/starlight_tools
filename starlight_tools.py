# -*- coding: utf-8 -*-
#
# This STARLIGHT library has several tools which can be employed to extract and 
# analyze STARLIGHT results. 
#
#                                                Costa-Duarte, M.V. 10/04/2013
#
#################################################################################
#
# Loading packages
#
import numpy as np
import bz2
import matplotlib.pyplot as plt

########################################################################################

def read_spectrum(infile):
    """
    This routine loads spec file. (lambda,flux,error,flag) 

                                       Costa-Duarte, M.V. - 10/04/2013
    """

    data = [line.strip() for line in open(infile)]
    l = []
    f = []
    error = []
    flag = []
    for line in data:
        p = line.split()
        if p[0] <> '#':
            l.append(float(p[0]))
            f.append(float(p[1]))
            error.append(float(p[2]))
            flag.append(int(p[3]))
    l = np.array(l)
    f = np.array(f)
    error = np.array(error)
    flag = np.array(flag)
    return l, f, error, flag

########################################################################################

def read_spectrum_bz2(infile):
    """
    This routine reads/loads a spectrum in BZ2 extention. (lambda,flux,error,flag) 

                                       Costa-Duarte, M.V. - 10/04/2013
    """

    bz_file = bz2.BZ2File(infile)
    data = bz_file.readlines()

    l = []
    f = []
    error = []
    flag = []
    for line in data:
        p = line.split()
        if p[0] <> '#':
            l.append(float(p[0]))
            f.append(float(p[1]))
            error.append(float(p[2]))
            flag.append(int(p[3]))
    l = np.array(l)
    f = np.array(f)
    error = np.array(error)
    flag = np.array(flag)
    return l,f,error,flag

########################################################################################

def make_aid(plate, mjd, fiber, DR):
    """
    This routine makes the AID (DR7 or DR9/DR10/BOSS)

                                       Costa-Duarte, M.V. - 10/04/2013
    """
    if len(plate) == len(mjd) & len(plate) == len(fiber):
      aid=[]
      str_plate=''
      str_mjd=''
      str_fiberID=''
      for i in range(len(plate)): 
        #      print 'plate, mjd, fiberID:',plate[i],mjd[i],fiber[i]

        if DR <> 'DR7': # BOSS aid -> fiberID with 4 digits
        # Plate
            if int(plate[i]) < 10: # 1 digit
                str_plate = '000' + str(int(plate[i]))
            if int(plate[i]) < 100 and plate[i] >= 10: # 2 digit
                str_plate = '00' + str(int(plate[i]))
            if int(plate[i]) < 1000 and plate[i] >= 100: # 3 digit
                str_plate = '0' + str(int(plate[i]))
            if int(plate[i]) >= 1000: # 4 digit
               str_plate = str(int(plate[i]))
        # fiberID 
            if int(fiber[i]) < 10: # 1 digit
               str_fiber='000' + str(int(fiber[i]))
            if int(fiber[i]) >= 10 and fiber[i] < 100: # 2 digits
               str_fiber='00' + str(int(fiber[i]))
            if int(fiber[i]) >= 100 and fiber[i] <= 999: # 3 digits
               str_fiber='0' + str(int(fiber[i]))
            if int(fiber[i]) == 1000: # 4 digits
               str_fiber= str(int(fiber[i]))
        else: # older DR -> fiberID with 3 digits
        # Plate
            if int(plate[i]) < 10: # 1 digit
                str_plate = '000' + str(int(plate[i]))
            if int(plate[i]) < 100 and plate[i] >= 10: # 2 digit
                str_plate = '00' + str(int(plate[i]))
            if int(plate[i]) < 1000 and plate[i] >= 100: # 3 digit
                str_plate = '0' + str(int(plate[i]))
            if int(plate[i]) >= 1000: # 4 digit
                str_plate = str(int(plate[i]))
        # fiberID 
            if int(fiber[i]) < 10: # 1 digit
                str_fiber='00' + str(int(fiber[i]))
            if int(fiber[i]) >= 10 and int(fiber[i]) < 100: # 2 digits
                str_fiber='0' + str(int(fiber[i]))
            if int(fiber[i]) >= 100 and int(fiber[i]) <= 999: # 3 digits
                str_fiber= str(int(fiber[i]))
        # MJD
        str_mjd = str(int(mjd[i]))
        aid.append(str_plate + '.' + str_mjd + '.' + str_fiber) 
        #print i, plate[i], mjd[i], fiber[i], str_plate, str_mjd, str_fiber
        #if i > 100: exit()
    return aid

########################################################################################

def read_synthesis(infile,dl):
    """
    This routine reads spectral synthesis results and calculate ages and metallicities (STARLIGHTv4).

                                                Costa-Duarte, M.V. - 11/04/2013
    """

#    Read the infile
    lines = [line.strip() for line in open(infile)]
#
    flag_synthesis_ok = 0
    if len(lines) > 3: # Some error and STARLIGHT could not fit it...
        nel_base = lines[9].split()[0]                                             # Number of elements of the BASE
        SN = float(lines[30].split()[0])                                           # sign-to-noise ratio
        sum_x = float(lines[52].split()[0])                                        # Sum of the light vector 
        if dl > 0.: # Given luminosity distance -> True Mass
            Mini = np.log10(float(lines[54].split()[0]) * 0.312728 * dl * dl)      # Initial stellar mass
            Mcor = np.log10(float(lines[55].split()[0]) * 0.312728 * dl * dl)      # Corrected stellar mass 
        else: # Do not have distance...
            Mini = np.log10(float(lines[54].split()[0]))                           # Initial stellar mass
            Mcor = np.log10(float(lines[55].split()[0]))                           # Corrected stellar mass 
        v0 = float(lines[57].split()[0])                                           # v0
        sigma = float(lines[58].split()[0])                                        # velocity dispersion
        AV =  float(lines[59].split()[0])                                          # Extinction parameter
        Nl =  float(lines[38].split()[0])                                          # # of lambdas
        Nl_clipped =  float(lines[40].split()[0])                                  # # of lambdas clipped
        frac_clipped = Nl_clipped / Nl
        chi2_red =  float(lines[49].split()[0])                                  # # of lambdas clipped

        # Reading light/mass vectors   
        x = np.zeros(int(nel_base))
        mu = np.zeros(int(nel_base))
        age = np.zeros(int(nel_base))
        Z = np.zeros(int(nel_base))
        for i in range(int(nel_base)):
            x[i] = lines[63+i].split()[1]    
            mu[i] = lines[63+i].split()[3]  
            age[i] = lines[63+i].split()[4]  
            Z[i] = lines[63+i].split()[5]
        #    print i+1,x[i],mu[i],str(age[i]),np.log10(age[i])
        #
        # Calculate average ages/metallicities 
        agelight = np.dot(np.log10(age),x)/sum(x)   
        agemass = np.dot(np.log10(age),mu)/sum(mu)   
        zlight = np.dot(np.log10(Z),x)/sum(x)    
        zmass = np.dot(np.log10(Z),mu)/sum(mu)
    else:
        flag_synthesis_ok = -99999.
        nel_base = -99999.
        SN = -99999.
        Mcor = -99999.
        Mini = -99999.
        AV = -99999.
        agemass = -99999.
        agelight = -99999.
        zmass = -99999.
        zlight = -99999.
        v0 = -99999.
        sigma = -99999. 
        chi2_red = -99999.

    return nel_base, SN, Mcor, Mini, AV, agemass, agelight, zmass, zlight, v0, sigma, frac_clipped, chi2_red, flag_synthesis_ok

###############################################################################################

def read_synthesis_spec(infile):
    """
    This routine reads the spectral synthesis fitting (STARLIGHTv4 format).

                                                          Costa-Duarte, M.V.- 10/04/2013
    """
#    Read the infile
    lines = [line.strip() for line in open(infile)]
# 
    nel_base = int(lines[9].split()[0])    # Number of elements of the BASE
    fobsnorm = float(lines[25].split()[0])    # Normalization factor of observed flux
#
#    if starlight_version == 4: # STARLIGHTv4
    n0 = 80 + 3 * nel_base
#    if starlight_version == 5: # STARLIGHTv5
#        n0 = 84 + 4 * nel_base
    nlambda = int(lines[n0].split()[0])
    l = np.zeros(nlambda)
    fobs = np.zeros(nlambda)
    fsyn = np.zeros(nlambda)
    wei = np.zeros(nlambda)
    for i in range(nlambda):
        l[i] =  float(lines[n0+1+i].split()[0])
        fobs[i] = fobsnorm * float(lines[n0+1+i].split()[1])
        fsyn[i] = fobsnorm * float(lines[n0+1+i].split()[2])
        wei[i] = fobsnorm * float(lines[n0+1+i].split()[3])
    return l, fobs, fsyn, wei

########################################################################################

def norm_spec(l,f,lambda_norm_i,lambda_norm_f):
    """
    Calculate the normalization factor
                                        Costa-Duarte, M.V. - 10/04/2013
    """
    l = np.array(l,dtype=np.float32)
    f = np.array(f,dtype=np.float32)
    idx_l = [(l >= lambda_norm_i) & (l <= lambda_norm_f)]
    fobs_norm = np.average(f[idx_l])
    return fobs_norm
############################################################################
def Cardelli_RedLaw(l,R_V=3.1):

    '''

        from W.Schoenel's routine
 
        @summary: Calculates q(lambda) to the Cardelli, Clayton & Mathis 1989 reddening law. Converted to Python from STALIGHT.
        @param l: Wavelenght vector (Angstroms)
        @param R_V: R_V factor. Default is R_V = 3.1.

    '''

#     q = A_lambda / A_V for Cardelli et al reddening law
#     l = lambda, in Angstrons
#     x = 1 / lambda in 1/microns
#     q = a + b / R_V; where a = a(x) & b = b(x)
#     Cid@INAOE - 6/July/2004
#
    a = np.zeros(np.shape(l))
    b = np.zeros(np.shape(l))
    F_a = np.zeros(np.shape(l))
    F_b = np.zeros(np.shape(l))
    x = np.zeros(np.shape(l))
    y = np.zeros(np.shape(l))
    q = np.zeros(np.shape(l))
#
    for i in range(0,len(l)):
     x[i]=10000. / l[i]
     y[i]=10000. / l[i] - 1.82
#
#
#     Far-Ultraviolet: 8 <= x <= 10 ; 1000 -> 1250 Angs 
    inter = np.bitwise_and(x >= 8, x <= 10)

    a[inter] = -1.073 - 0.628 * (x[inter] - 8.) + 0.137 * (x[inter] - 8.)**2 - 0.070 * (x[inter] - 8.)**3
    b[inter] = 13.670 + 4.257 * (x[inter] - 8.) - 0.420 * (x[inter] - 8.)**2 + 0.374 * (x[inter] - 8.)**3

#     Ultraviolet: 3.3 <= x <= 8 ; 1250 -> 3030 Angs 

    inter =  np.bitwise_and(x >= 5.9, x < 8)
    F_a[inter] = -0.04473 * (x[inter] - 5.9)**2 - 0.009779 * (x[inter] - 5.9)**3
    F_b[inter] =  0.2130 * (x[inter] - 5.9)**2 + 0.1207 * (x[inter] - 5.9)**3
    
    inter =  np.bitwise_and(x >= 3.3, x < 8)
    
    a[inter] =  1.752 - 0.316 * x[inter] - 0.104 / ((x[inter] - 4.67)**2 + 0.341) + F_a[inter]
    b[inter] = -3.090 + 1.825 * x[inter] + 1.206 / ((x[inter] - 4.62)**2 + 0.263) + F_b[inter]

#     Optical/NIR: 1.1 <= x <= 3.3 ; 3030 -> 9091 Angs ; 
    inter = np.bitwise_and(x >= 1.1, x < 3.3)
    
    a[inter] = 1.+ 0.17699 * y[inter] - 0.50447 * y[inter]**2 - 0.02427 * y[inter]**3 + 0.72085 * y[inter]**4 + 0.01979 * y[inter]**5 - 0.77530 * y[inter]**6 + 0.32999 * y[inter]**7
    b[inter] = 1.41338 * y[inter] + 2.28305 * y[inter]**2 + 1.07233 * y[inter]**3 - 5.38434 * y[inter]**4 - 0.62251 * y[inter]**5 + 5.30260 * y[inter]**6 - 2.09002 * y[inter]**7


#     Infrared: 0.3 <= x <= 1.1 ; 9091 -> 33333 Angs ; 
    inter = np.bitwise_and(x >= 0.3, x < 1.1)
    
    a[inter] =  0.574 * x[inter]**1.61
    b[inter] = -0.527 * x[inter]**1.61
    
    q = a +  (b / R_V)

    return q


def calc_redlaw(l, R_V,redlaw):
    if(redlaw == 'CCM'): return Cardelli_RedLaw(l, R_V)

########################################################################################

def calc_SN(l,f,flag,lambda_SN_i,lambda_SN_f):
    """
    This routine calculates the S/N ratio of a spectrum within a wavelength range.. 

                                             Costa-Duarte, M.V. - 10/04/2013
    """

    l = np.array(l,dtype=np.float32)
    f = np.array(f,dtype=np.float32)
    flag = np.array(flag,dtype=np.float32)

    idx_l = [(l >= lambda_SN_i) & (l <= lambda_SN_f) & (flag <= 1.)]
    S = np.average(f[idx_l])
    N = np.std(f[idx_l])
    return S/N

########################################################################################

def make_spec_from_synthesis(file_syn, path_ssp,path_base):
    """
    This routine builds the spectrum from synthesis spectrum results (STARLIGHTv4)

                                                    Costa-Duarte, M.V. - 01/05/2013
    """
    #   Read the infile

    lines = [line.strip() for line in open(file_syn)]

    file_base =  str(lines[6].split()[0])        # Base file
    nel_base = lines[9].split()[0]               # Number of elements of the BASE
    lambda_norm_i =  float(lines[23].split()[0]) # Normalization initial lambda
    lambda_norm_f =  float(lines[24].split()[0]) # Normalization final lambda
    fobsnorm = float(lines[25].split()[0])       # Normalization factor of observed flux
    sum_x =  float(lines[52].split()[0])         # Sum of the light vector 
    A_V =  float(lines[59].split()[0])            # Extinction parameter

    #   Light vector

    x = np.zeros(int(nel_base))
    for i in range(int(nel_base)):
        x[i] = lines[63+i].split()[1] 
    #print('sum x=',sum(x))
    x = x / 100.
    #print('sum x / 100=',sum(x))
    #exit()

    #   Reading the spectral base 

    lines = [line.strip() for line in open(path_base+file_base)]
    file_ssp = []
    for i in range(int(lines[0].split()[0])+1):
        if i > 0: # 1st line - number of SSPs
            file_ssp.append(str(lines[i].split()[0]))

    #   Read SSP spectra and make spectrum

    for i in range(len(file_ssp)):
       f = open(path_ssp + file_ssp[i])
       data = f.readlines()[6:]
       f.close()
       # lambda and flux of SSPs
       l_ssp = []
       f_ssp = []
       for line in data:
           p = line.split()
           l_ssp.append(float(p[0]))
           f_ssp.append(float(p[1]))
       l_ssp = np.array(l_ssp)
       f_ssp = np.array(f_ssp)
       if i == 0:
               f_syn = np.zeros(len(l_ssp))
               l_syn = l_ssp
       fobsnorm_ssp = float(norm_spec(l_ssp, f_ssp, lambda_norm_i, lambda_norm_f))
       f_syn += x[i] * f_ssp / fobsnorm_ssp
       #print i, x[i] 

    #   Add extinction (it is galaxy extinction, not Galactic one)

    q = calc_redlaw(l_syn, 3.1,  'CCM')
    A_lambda = A_V * q
    f_syn = f_syn * 10. ** (- 0.4 * A_lambda)

    #   Normalization to ensure the flux calibration 

    fobsnorm0 = float(norm_spec(l_syn,f_syn,  lambda_norm_i, lambda_norm_f))
    f_syn = (f_syn / fobsnorm0) * fobsnorm

    return l_syn, f_syn

#########################################################################################

def interpol_spec(l0,f0,error0,flag0,li,lf,dl):
    """
    This routines interpolates the spectrum with new resolution (dl) and
    keeping the flags and errors. 
                                              Costa-Duarte, M.V. - 03/05/2013
    """
    # Resampling  of Flux and Error_flux

    l = np.arange(li,lf,dl) # New lambda-vector
    f = np.interp(l, l0, f0, left = 0.0, right = 0.0)
    error = np.interp(l, l0, error0, left = 0.0, right = 0.0)

    # Resampling  of Flag

    flag = np.zeros(len(l)) # Resampling -> flag
    for j in range(len(l)):
        idx = (np.abs(l0-l[j])).argmin()
        flag[j] = flag0[idx]

    flag[f == 0.] = 99  # Flux = 0. are flagged as 99 (not considered in synthesis!)    

    return l,f,error,flag

#########################################################################################

def generate_popvec(x):
    """

    This routines generates a population vector (light or mass vector) in order to 
    simulate a galaxy spectrum. 
                                              Costa-Duarte, M.V. - 10/04/2013
    """

    x[:] = 0.
    if len(x) == 1: # Only 1 component 
        x[0] = 1.
    else: # More than 1 component    

        sum_x = 0.
        flag_idx = np.zeros(len(x))

        while len(flag_idx[(flag_idx == 1)]) < len(x):
            idx = int(float(len(x)) * np.random.uniform(low=0.,high=1.))
            if flag_idx[idx] == 0.:
                if len(flag_idx[(flag_idx == 1)]) == len(x)-1: 
                    x[idx] = (1. - sum_x)
                    flag_idx[idx] = 1.

                else:

                    x[idx] = (1. - sum_x) * abs(np.random.uniform(low=0.,high=1.))
                    sum_x = sum_x + x[idx]
                    flag_idx[idx] = 1.
    return x
#########################################################################################

def mask_spec(l,f,error,flag,file_mask,l_out):
    """
    This routine masks a spectrum by using a file_mask and resamples it after discarting
    masked regions + bad pixels.
    flag=99 # bad pixel for spectral synthesis
                                              Costa-Duarte, M.V. - 07/07/2014
    """
    # Numpy arrays
    l0 = np.array(l)
    f0 = np.array(f)
    error0 = np.array(error)
    flag0 = np.array(flag)

    # 1st line
    lines = [line.strip() for line in open(file_mask)]
    lresam = l_out # New lambda-vector

    # 
    n_mask = int(lines[0].split()[0])    # Number lambda windows of the MASK
    l_mask_i = []
    l_mask_f = []
    for i in range(n_mask):    
        l_mask_i.append(float(lines[i+1].split()[0])) # lambda_mask_initial
        l_mask_f.append(float(lines[i+1].split()[1])) # lambda_mask_final
    l_mask_i = np.array(l_mask_i)
    l_mask_f = np.array(l_mask_f)

    # Deleting masked regions

    flag_mask = np.zeros(len(l0))
    for i in range(n_mask):    
        idx = [(l0 >= l_mask_i[i]) & (l0 <= l_mask_f[i])]
        flag_mask[idx] = 99999.
    idx_flag = [(flag_mask <> 99999.) & (flag0 <= 1)] # Masked regions + bad pixels

    # Resampling of Flux and Error_flux

    fresam = np.interp(lresam,l0[idx_flag],f0[idx_flag],left = 0.0,right = 0.0)
    error_resam = np.interp(lresam,l0[idx_flag],error0[idx_flag], left = 0.0, right = 0.0)

    # Resampling of Flag

    flag_resam = np.zeros(len(lresam)) # Resampling -> flag
    for j in range(len(lresam)):
        idx = (np.abs(l0-lresam[j])).argmin()
        flag_resam[j] = flag0[idx]
    flag_resam[fresam == 0.] = 99  # Flux = 0. are flagged as 99 (not considered in synthesis!)

    return lresam,fresam,error_resam,flag_resam

################################################################################

def calc_D4000(l,f,error,lambda_D4000_blue_i,lambda_D4000_blue_f,lambda_D4000_red_i,lambda_D4000_red_f):
    """
    This routine calculates the D4000-break of a spectrum, following the
    Balogh+99 (http://adsabs.harvard.edu/abs/1999ApJ...527...54B) formalism. 

                                             Costa-Duarte, M. V. - 14/04/2015
    """
    # C_red
    C_red=0.
    C_red_error=0.
    n_red=0.
    for i in range(len(l)):
        if l[i] >= lambda_D4000_red_i and l[i] <= lambda_D4000_red_f and f[i] <> 0.: 
            C_red += f[i] * l[i] * l[i]
            C_red_error += (l[i] * l[i] * error[i]) **2
            n_red += 1.

    # C_red error
    C_red_error=(1./n_red) * np.sqrt(C_red_error)

    # C_blue
    C_blue=0.
    C_blue_error=0.
    n_blue=0.
    for i in range(len(l)):
        if l[i] >= lambda_D4000_blue_i and l[i] <= lambda_D4000_blue_f and f[i] <> 0.: 
            C_blue += f[i] * l[i] * l[i]
            C_blue_error += (l[i] * l[i] * error[i]) **2
            n_blue += 1.

    # C_blue_error
    C_blue_error=(1./n_blue) * np.sqrt(C_blue_error)

    # calculate D4000
    if n_blue > 0. and n_red > 0.: 
        C_blue = (C_blue/n_blue) 
        C_red = (C_red/n_red) 
        D4000 = C_red/C_blue
        D4000_error = np.sqrt((C_red_error / C_blue)**2 + ((C_red * C_blue_error)**2 / C_blue**4))
    else:
        D4000 = -99999.
        D4000_error = -99999.

    return D4000,D4000_error

########################################################################################

def load_ssp_bc03(infile):
    """

    This routine loads BC03 spectrum. 

                                     Costa-Duarte, M.V. - 13/04/2014
    """
    data = [line.strip() for line in open(infile)]
  
    l = []
    f = []
    for i in range(6,len(data)): # 6 lines header
        l.append(float(data[i].split()[0]))
        f.append(float(data[i].split()[1]))
  
    f = np.array(f)
    l = np.array(l)
  
    return l,f

#########################################################################################

def add_noise_SSP(l,f,l_error,vec_error,SN):
    """
    This routine adds some noise to a spectrum by considering its pixel error array.

                  f_noisy = f * (1 + gasdev * error_vector / SN)

    ATTENTION: It's necessary to BRING the error vector to the flux amplitude (respect the SN ratio) 
    before using this routine.

                                                Costa-Duarte - 17/12/2014
    """
    #
    f_out = np.zeros(len(l))
    for j in range(len(l)):
        idx_l_error = [(l[i] == l_error)] # Same lambda for l and l_error
        if len(l_error[idx_l_error]) == 1: # 1 match of lambdas
            f_out[j] = f[j] * (1. + np.random.normal(0.,1.,len(l)) * vec_error[idx_l_error] / SN)
        if l[j] < min(l_error): # l[j] is out of error range -> Consider 1st array element
            f_out[j] = f[j] * (1. + np.random.normal(0.,1.,len(l)) * vec_error[0] / SN) 
        if l[j] > max(l_error): # l[j] is out of error range -> Consider 1st array element
            f_out[j] = f[j] * (1. + np.random.normal(0.,1.,len(l)) * vec_error[len(vec_error)-1] / SN) 
    return f_out

#########################################################################################

def mask2flag(mask):
    """
    This routine converts mask vector from SDSS fits file of spectra to mask 
    for STARLIGHT. 


    from ABILIO's code...

       FLAGS:

       0: OK pixel
       1: emission line
       > 1: problems...



                                               Costa-Duarte, M.V., 09/02/2014
    """
  
    r=np.zeros(10)    
    r[0] = 262144      # FULLREJECT
    r[1] = 524288      # PARTIALREJ
    r[2] = 4194304     # NOSKY
    r[3] = 8388608     # BRIGHTSKY
    r[4] = 16777216    # NODATA
    r[5] = 33554432    # COMBINEREJ
    r[6] = 67108864    # BADFLUXFACTOR
    r[7] = 134217728   # BADSKYCHI
    r[8] = 268435456   # REDMONSTER
    r[9] = 1073741824  # EMLINE
  #
    nl=len(mask)
    flag=np.zeros(nl)
  #
    for i in range(0,nl):
        flag[i]=0 # pixel OK!
        if mask[i] >= r[9]: flag[i]=1 # EMLINE
        if mask[i] >= r[9]+r[0] and mask[i] < r[9]+r[1]: flag[i]=2 #EMLINE + FULLREJ
        if mask[i] >= r[9]+r[2] and mask[i] < r[9]+r[3]: flag[i]=3 #EMLINE + NOSKY
        if mask[i] >= r[9]+r[3] and mask[i] < r[9]+r[4]: flag[i]=4 #EMLINE + SKY
        if mask[i] >= r[9]+r[4] and mask[i] < r[9]+r[5]: flag[i]=5 #EMLINE + NODATA
        if mask[i] >= r[9]+r[7] and mask[i] < r[9]+r[8]: flag[i]=6 #EMLINE + BADSKY
        if mask[i] >= r[9]+r[7]+r[3] and mask[i] < r[9]+r[8]: flag[i]=7 #EMLINE + BADSKY + SKY
        if mask[i] >= r[0] and mask[i] < r[1]: flag[i]=8 #FULLREJECT
        if mask[i] >= r[2] and mask[i] < r[3]: flag[i]=9 # NOSKY
        if mask[i] >= r[3] and mask[i] < r[4]: flag[i]=10 #SKY
        if mask[i] >= r[3]+r[4] and mask[i] < r[5]: flag[i]=11 #SKY + NODATA
        if mask[i] >= r[4] and mask[i] < r[5]: flag[i]=12 #NODATA
        if mask[i] >= r[6] and mask[i] < r[7]: flag[i]=13 #BADFLUXFACTOR
        if mask[i] >= r[6]+r[4] and mask[i] < r[6]+r[5]: flag[i]=14 #BADFLUX + NODATA
        if mask[i] >= r[6]+r[3] and mask[i] < r[6]+r[4]: flag[i]=15 #BADFLUX + SKY
        if mask[i] >= r[7] and mask[i] < r[8]: flag[i]=16  #BADSKY
        if mask[i] >= r[7]+r[3] and mask[i] < r[7]+r[4]: flag[i]=17 #BADSKY + SKY
        if mask[i] >= r[7]+r[3]+r[0] and mask[i] < r[7]+r[3]+r[1]: flag[i]=18 #BADSKY + SKY + FULLREJ
        if mask[i] >= r[8] and mask[i] < r[9]: flag[i]=19 #REDMONSTER

    return flag

#########################################################################################

def idx_n_highest_values(a,N):
    """
    Index of the highest value in an array
    """
    return np.argsort(a)[::-1][:N]

#########################################################################################

def class_whan(xx,yy,EW_NII):
    """
    This routine classifies galaxies by using emission lines with WHAN diagram.
    (Cid Fernandes et al 2011)
                                                   Costa-Duarte - 12/11/2014
    """
    #
    class_gal = 0.
    # SF
    if xx <= -0.4 and yy > np.log10(3.):
        class_gal = 1.
    # sAGN
    if xx > -0.4 and yy > np.log10(6.):
        class_gal = 2.
    # wAGN
    if xx > -0.4 and yy >= np.log10(3.) and yy <= np.log10(6.):
        class_gal = 3.
    # retired
    if yy < np.log10(3.):
        class_gal = 4.
    # passive
    if yy <= np.log10(0.5) and EW_NII <= np.log10(0.5):
        class_gal = 5.
    # Any problem? NO classification?
    if class_gal == 0.:
        print 'PROBLEM - NO CLASSIFICATION'
        exit()
    return class_gal
#########################################################################################
def class_whan_condensed(xx, yy):
    """

    This routine classifies galaxies by using emission lines with WHAN diagram.
    (Cid Fernandes et al 2011)

    This diagram is condensed because we put s+wAGN and passive + retired in the same class

        SF - 1
        s+w AGN - 2
        passive + retired - 3

                                                   Costa-Duarte - 12/11/2014
    """
    #
    class_gal = 0.
    # SF (class=1)
    if xx <= -0.4 and yy > np.log10(3.):
        class_gal = 1.
    # sAGN (class=2)
    if xx > -0.4 and yy > np.log10(6.):
        class_gal = 2.
    # wAGN (class=2)
    if xx > -0.4 and yy > np.log10(3.) and yy <= np.log10(6.):
        class_gal = 2.
    # retired (class=3)
    if yy <= np.log10(3.):
        class_gal = 3.
    # passive (class=3)
    if yy <= np.log10(0.5):
        class_gal = 3.
        
    #print xx, yy, class_gal

    # Any problem? NO classification?
    if class_gal == 0.:
        print 'PROBLEM - NO CLASSIFICATION'
        exit()
    return class_gal

#########################################################################################

def dist_sky_arcsec(ra1,dec1,ra2,dec2):
    """
    This routine calculates the sky distance in arcsec (Haversine equation). 

                                                 Costa-Duarte, M.V. - 10/12/2014
    """
    d2r=np.pi/180.                       # degree2radian
    arcs2r = np.pi/(180. * 3600.) # arcsec2radian
    return np.arccos(np.sin(dec1*d2r)*np.sin(dec2*d2r)+np.cos(dec1*d2r)*np.cos(dec2*d2r)*np.cos(ra1*d2r-ra2*d2r)) / arcs2r

#########################################################################################

def calc_SFR(infile,dl,tmax):
    """
    This routine calculates the recent (t<tmax) SFR according to the light vector 
    of the STARLIGHT output. The luminosity distance (DL) is necessary in order to 
    estimate the stellar mass (M*) and consequently the SFR (Msun/yr).

                                        Costa-Duarte, M.V. 15/11/2014
    """
#    Read the infile
    lines = [line.strip() for line in open(infile)]
#
    nel_base = lines[9].split()[0]                    # Number of elements of the BASE
    Mini = np.log10(float(lines[54].split()[0]))      # Initial stellar mass
    Mini *= 0.312728 * dl * dl                        # distance-correction(in Mpc)
#
    x = np.zeros(int(nel_base))
    mu = np.zeros(int(nel_base))
    age = np.zeros(int(nel_base))
    Z = np.zeros(int(nel_base))
    for i in range(int(nel_base)):
        x[i] = lines[63+i].split()[1]    
        mu[i] = lines[63+i].split()[3]  
        age[i] = lines[63+i].split()[4]  # in lookback time years
        Z[i] = lines[63+i].split()[5]

    # Mass fraction of recent (t<tmax) star formation

    idx = [(age < tmax)]
    if len(age[idx]) > 0:
        frac_mass_recent_sf = sum(mu[idx])/sum(mu)
    else:
        frac_mass_recent_sf = 0.

    # SFR = M_ini *(recent, t<tmax) / tmax

    SFR = (frac_mass_recent_sf * Mini) / tmax # Msun/yr

    return SFR

########################################################################################

def read_eline(infile,l_eline):
    """
    This routine reads the ELINE files (Abilio's code to measure emission lines).
    Given an array of lambdas of emission lines, they select the values of
    flux, EW, velocity dispersion, velocity (and their errors) and signal-to-noise
    of the emission lines. 
                                        Costa-Duarte, M.V., 01/12/2014

    Header Abilio's code for eline outfiles:

        lline(k),Ie(k),sigIe(k),EWe(k),sigEWe(k),
        sigma(k),sig_sigma(k),v0_e(k),sig_v0_e(k),
        SNe(k),fcontinuum(k),sigfcontinuum(k),chisqr(k)

    """
    #    Read the infile
    lines0 = [line.strip() for line in open(infile)]
    #
    l = []
    F = []
    F_sig = []
    EW = []
    EW_sig = []
    veldisp = []
    veldisp_sig = []
    v0 = []
    v0_sig = []
    SN = []
    #
    if len(lines0) > 0:
        lines0 = np.array(lines0)
        l0 = []
        for line in lines0:
            p = line.split()
            if p[0] <> '#':
                l0.append(float(p[0]))
        l0 = np.array(l0)
        #
        for j in range(len(l0)):
            for k in range(len(l_eline)):
                if l_eline[k] == float(lines0[j].split()[0]):  
                    l.append(float(lines0[j].split()[0]))
                    F.append(float(lines0[j].split()[1]))
                    F_sig.append(float(lines0[j].split()[2]))
                    EW.append(float(lines0[j].split()[3]))
                    EW_sig.append(float(lines0[j].split()[4]))
                    veldisp.append(float(lines0[j].split()[5]))
                    veldisp_sig.append(float(lines0[j].split()[6]))
                    v0.append(float(lines0[j].split()[7]))
                    v0_sig.append(float(lines0[j].split()[8]))
                    SN.append(float(lines0[j].split()[9]))
    else:
        for k in range(len(l_eline)):
            l.append(-99999.)
            F.append(-99999.)
            F.append(-99999.)
            EW.append(-99999.)
            EW_sig.append(-99999.)
            veldisp.append(-99999.)
            veldisp_sig.append(-99999.)
            v0.append(-99999.)
            v0_sig.append(-99999.)
            SN.append(-99999.)

    l = np.array(l)
    F = np.array(F)
    F_sig = np.array(F_sig)
    EW = np.array(EW)
    EW_sig = np.array(EW_sig)
    veldisp = np.array(veldisp)
    veldisp_sig = np.array(veldisp_sig)
    v0 = np.array(v0)
    v0_sig = np.array(v0_sig)
    SN = np.array(SN)

    return l,F,F_sig,EW,EW_sig,veldisp,veldisp_sig,v0,v0_sig,SN   

###################################################################################

def check_bad_pixels(l, f, error, flag, l_eline, v0, sigma, nsigma):
    """
    This routine checks the fraction of bad pixels at the regions 
    of emission lines. For each eline, a fraction of bad pixels is output.

                                              Costa-Duarte, M.V. - 01/12/2014
    """
    # 
    c = 299792.458 # light speed (km/s)

    frac_bad_pixel = np.zeros(len(l_eline)) # Declare fraction of bad pixel array for each eline

    for i in range(len(l_eline)): # all emission lines array
    
        # Define spectral range of emission line

        if v0[i] > -900. and sigma[i] > -900.:

            # central wavelength
            lambda0 = l_eline[i] / (1. + v0[i] / c)                       

            # central wavelength - nsigma * sigma (red)
            lambda_red = lambda0 / (1. + (v0[i] - nsigma * sigma[i]) / c)

            # central wavelength + nsigma * sigma (blue)
            lambda_blue = lambda0 / (1. + (v0[i] + nsigma * sigma[i]) / c)

            # Counting number of pixels

            # number of bad pixels (flag > 1) in the spectral region
            npixel_bad = float(len(l[(l >= lambda_blue) & (l <= lambda_red) & (flag > 1)])) 
            npixel_tot = float(len(l[(l >= lambda_blue) & (l <= lambda_red)])) # total number of pixels in the region

            print 'lambda0=',lambda0
            print 'lambda blue and red=',lambda_blue,lambda_red
            print 'npixel=',npixel_tot,npixel_bad

            if npixel_tot > 0.: # Spectral coverage includes this eline 
                frac_bad_pixel[i] = npixel_bad / npixel_tot

            if npixel_tot <= 0.: # No pixels at the eline region
                frac_bad_pixel[i] = -99998.
        else:

            frac_bad_pixel[i] = -99999. # No measurement of the emission line
        #
    return frac_bad_pixel
#########################################################################################
def kcorrect_omill_sdss(mg,mr,redshift):
     """
      This routine calculates the k-correction by using g, r and redshift 
      according to the linear interpolation/fitting from O'Mill et al. (2011), 413, 1395.
      The formalism (coefficients) used here follows the same presented in the paper.
   
                                    Costa-Duarte, M.V. - 03/12/2014
     """
     # From Table 2 of O'Mill et al (2011)
     #
     # coefficient A
     a_A = [2.956,3.070,1.771,0.538,0.610]
     a_A_sig = [0.070,0.165,0.032,0.085,0.045]
     b_A = [-0.100,0.727,-0.529,-0.075,-0.064]
     b_A_sig = [0.034,0.117,0.023,0.079,0.034]
     # 
     # coefficient B
     a_B = [-0.299,-0.313,-0.179,-0.027,-0.061]
     a_B_sig = [0.009,0.021,0.005,0.013,0.007]
     b_B = [-0.095,-0.173,-0.048,-0.120,-0.106]
     b_B_sig = [0.004,0.015,0.003,0.012,0.005]
     #
     if isinstance(mr,(np.ndarray)) == True:
         kc = np.zeros(len(a_A) * len(mr)).reshape(len(a_A),len(mr))
     if isinstance(mr,(np.ndarray)) == False:
         kc = np.zeros(len(a_A))
     #
     for i in range(len(a_A)):
          kc[i] =  (a_A[i] * (mg - mr) + b_A[i]) * redshift + (a_B[i] * (mg - mr) + b_B[i])
     return kc
########################################################################################

def mag2Mabs(mag,dl):
    """
    Calculate the Absolute magnitude, given magnitude and luminosity distance.
    mag - magnitude corrected by extinction, offset and k-correction
        mag = m(raw) - extinction + offset + kc
    dl - luminosity distance (in Mpc)

                                        Costa-Duarte, M.V. - 03/12/2014
    """
#
    Mabs = mag - 5. * np.log10(dl) - 25.
    return Mabs

########################################################################################

def flux2magAB(F_l_eff,F_l_eff_error,l_eff):
    """

     This routine transform flux (in units of 10^-17 erg/s/A/cm2) into AB magnitudes.
     The value of l_eff is calculated according to the routine "synthetic_photometry".  

                                                 Costa-Duarte - 10/12/2014
    """

    # light speed in Angstroms/s  
    c=2.99792458e18

    # magAB - F_l_eff in 10^-17 erg/s/A/cm2
    magAB = -2.5 * np.log10((F_l_eff * l_eff ** 2) / c) - 6.1

    # magAB error: sqrt( (-2.5) ** 2) * (1/(F_l * ln(10)) * F_l_eff_error  
    # magAB_error = np.sqrt((-2.5) ** 2) * (1. / (F_l_eff * np.log(10.)) * F_l_eff_error
    magAB_error = -99999.

    return magAB,magAB_error
########################################################################################
def load_sdss_filter(infile,airmass):
    """
    This routine loads the transmission filter files of SDSS.
    All '#' are considered headers and 2 airmass values are avaiable (0. and 1.3).

                                             Costa-Duarte - 10/12/2014
    """

    if airmass <> 0. and airmass <> 1.3:
        print 'Airmass NOT avaiable!'
        print 'SDSS Airmass = 0 or 1.3!'
        exit() 

    # Load infile

    data = [line.strip() for line in open(infile)]
    l_filter = []
    s_filter = []
    for line in data:
        p = line.split()
        if p[0] <> '#':
            l_filter.append(float(p[0]))
            if airmass == 0.:
                s_filter.append(float(p[3])) # extended source and airmass = 0.
            if airmass == 1.3:
                s_filter.append(float(p[2])) # extended source and airmass = 1.3

    l_filter = np.array(l_filter)
    s_filter = np.array(s_filter)

    return l_filter,s_filter
########################################################################################
def synthetic_photometry(l_spec0,f_spec0,error_spec0,flag_spec0,l_filter0,s_filter0,dlambda):
    """
    This routine calculates the convolution of a pass-band filter and a spectrum.
    The resulting flux consists of a synthetic photometry as defined below:
 
    F_l_eff = integral(F_l * S_l * dl)/integral(S_l * dl) 

    l_eff = integral(S_l * l * dl)/integral(S_l * dl)
  
    Equations from Hyperz's Manual:

    www.bo.astro.it/~micol/Hyperz/old_archive/hyperz_manual1.2.ps.gz

                                                        Marcus - 10/12/2014
    """

    # Put the observed spectrum at the spectral resolution (dlambda)

    l_spec,f_spec,error_spec,flag_spec = interpol_spec(l_spec0, f_spec0, error_spec0, flag_spec0, 
                                         np.round(min(l_spec0)), np.round(max(l_spec0)), dlambda)

    # Put the filter transmission at the spectral resolution (dlambda)

    l_filter, s_filter, error_filter,flag_filter = interpol_spec(l_filter0, s_filter0, 
                                                   0.1 * s_filter0, np.zeros(len(l_filter0)), 
                                                   np.round(min(l_spec0)), np.round(max(l_spec0)), dlambda)

    Fl_Sl_dl = 0.
    Fl_Sl_dl_error = 0.
    Sl_dl = 0.
    Sl_l_dl = 0.
    dl = (l_spec[1:] - l_spec[0:len(l_spec)-1])
    l_bins = 0.5 * (l_spec[1:] - l_spec[0:len(l_spec)-1])

    # Trapezoidal Integration
    # Calculate the mean flux inside the bins 

    Fl_bins = 0.5 * (f_spec[1:] - f_spec[0:len(f_spec)-1])
    Sl_bins = 0.5 * (s_spec[1:] - s_spec[0:len(s_spec)-1])
    error_bins = 0.5 * (error_spec[1:] - error_spec[0:len(error_spec)-1])

    #
    # Integration of Fl * Sl * dl
    Fl_Sl_dl = sum(Fl_bins * Sl_bins * dl)

    # Error of Fl * Sl * dl: sqrt(sum d(Fl * Sl * dl)/dF_l)**2 * error_F_l)**2) = sqrt(sum (Sl * dl * error_F_l)**2) 
    Fl_Sl_dl_error = sum((Sl_bins * dl * error_bins) ** 2)

    # Integration of Sl * delta_l - NO ERROR in S_l and l
    Sl_dl = sum(Sl_bins * dl)

    # Integrartion of Sl * dl
    Sl_l_dl = sum(Sl_bins * l_bins * dl)

    if Fl_Sl_dl <> 0. and Sl_dl <> 0.:
        # Effective Flux
        Fl_eff = Fl_Sl_dl / Sl_dl
        # Effective lambda
        l_eff = Sl_l_dl / Sl_dl
        # Error in effective flux - sqrt((1/Sl_dl) ** 2 * Fl_Sl_l_error)
        Fl_Sl_dl_error = np.sqrt(Fl_Sl_dl_error) # sqrt of sum
        Fl_eff_error = (Fl_Sl_dl_error / Sl_dl) 
        # Getting AB magnitude         
        magAB, magAB_error = flux2magAB(Fl_eff, Fl_eff_error, l_eff)
        #
        flag_output = 0
    else:
        Fl_eff = -99999.
        Fl_eff_error = -99999.
        magAB = -99999.
        magAB_error = -99999.
        l_eff = -99999.

    return Fl_eff, Fl_eff_error, l_eff, magAB, magAB_error, flag_output
########################################################################################
def load_jplus_filter(infile):
    """
    This routine loads the transmission filter files of JPLUS.
    All '#' are considered headers and is loaded the 
    Transmission + CCD efficiency (3rd column!).

                                             Costa-Duarte - 10/12/2014
    """

    # Load filter transmission

    data = [line.strip() for line in open(infile)]
    l_filter = []
    s_filter = []
    for line in data:
        p = line.split()
        if p[0] <> '#':
            l_filter.append(float(p[0]))
            s_filter.append(float(p[1])) # T + CCD eff

    #
    l_filter = np.array(l_filter)
    s_filter = np.array(s_filter)

    return l_filter,s_filter

########################################################################################

def load_jplus_set_filters(infile_list, path_filters, dl):
    """
    This routine loads the transmission filter files of JPLUS.
    All '#' are considered headers.

                                             Costa-Duarte - 10/12/2014
    """

    # Load list of filters

    file_filters = [line.strip() for line in open(path_filters + infile_list)]

    
    nline = np.zeros(len(file_filters)) # number of lambdas of each filter
    for i in range(len(file_filters)):
        l, s = load_jplus_filter(path_filters + file_filters[i])
        l2 = np.arange(float(int(min(l))), float(int(max(l))), dl) 
        nline[i] = len(l2)

    # Declare output matrices of lambda and QE

    s_filter_matrix = np.zeros(len(file_filters) * int(max(nline))).reshape(len(file_filters), int(max(nline)))
    l_filter_matrix = np.zeros(len(file_filters) * int(max(nline))).reshape(len(file_filters), int(max(nline)))

    # Filling matrices with interpolated QEs in 1\AA

    for i in range(len(file_filters)):
        l, s = load_jplus_filter(path_filters + file_filters[i])
        l_new = np.arange(float(int(min(l))), float(int(max(l))), dl) 
        f_new = np.interp(l_new, l, s, left = 0.0, right = 0.0)
        s_filter_matrix[i][:len(l_new)] = f_new
        l_filter_matrix[i][:len(l_new)] = l_new

    # Plotting filters

#    for i in range(len(file_filters)):
#        idx = [(s_filter_matrix[i] > 0.)]
#        plt.plot(l_filter_matrix[i][idx], s_filter_matrix[i][idx]) 
#    plt.show()
#    exit()
    return l_filter_matrix, s_filter_matrix, file_filters

#########################################################################################
def BPT_lines(x,str_line):
    """
    This routine makes the Y vector according to the x-axis range (array) given 
    previously. 

    The lines calculated by several works are:

    1 - Stasinska+06
    2 - Kewley+01
    3 - Kauffmann+03 

                                                 Costa-Duarte - 17/12/2014
    """
    # Stasinska+06 line
    if str_line == 'stasinska':
        y = (-30.787 + 1.1358 * x + 0.27297 * x * x) * np.tanh(5.7409 * x) - 31.093
    # Kewley+01 line
    if str_line == 'kewley':
        y = 0.61 / (x - 0.47) + 1.19
    # Kauffmann+03 line
    if str_line == 'kauffmann':
        y = 0.61 / (x - 0.05) + 1.3
    return y
#########################################################################################
def class_BPT_kauffmann_kewley(x, y):
    """
    This routine classifies galaxies in BPT diagram according to Kauffmann et al (2003)
    and Kewley et al (2001). These galaxies are classified as:

    https://sites.google.com/site/agndiagnostics/home/bpt

    x = log([NII]/Ha) and y = log([OIII]/Hb)

    1 - Star-forming: below Kauffmann et al (2003)'s curve.
    2 - Composite: Between Kauffmann et al 2003's and Kewley et al (2001)'s curves.
    3 - AGNs: Above Kewley et al (2001)'s curve.


                                                 Costa-Duarte - 17/12/2014
    """

    y_kauffmann = 0.61 / (x - 0.05) + 1.3
    y_kewley = 0.61 / (x - 0.47) + 1.19

#    if x >= 0.05 and x < 0.47: print x, y, y_kewley, y_kauffmann

    #
    class_BPT = 0
    if y <= y_kauffmann and y < y_kewley and x < 0.47: class_BPT = 1 # SF
    if y <= y_kauffmann and x < 0.05: class_BPT = 1 # SF
    if y >= y_kauffmann and y < y_kewley: class_BPT = 2 # Composite
    if y < y_kewley and x > 0.05 and x < 0.47: class_BPT = 2 # Composite
    if y > y_kewley and y >= y_kauffmann: class_BPT = 3 # AGN
    if x > 0.47: class_BPT = 3 # AGN

    # At the extreme left of the diagram, these curves cross each other, then...(rarely!)

    if y > y_kewley: class_BPT = 3 # AGN
    #
    if class_BPT == 0:
         print 'PROBLEM BPT_kauffmann_kewley CLASSIFICATION!'
         exit()

    return class_BPT

############################################################################################
def condensed_vectors(infile):
    """
    This routine calculates the condensed vector according to CF+05.

                                   Costa-Duarte - 17/12/2014
    """

    # Read the infile
    lines = [line.strip() for line in open(infile)]
    nel_base = lines[9].split()[0]   

    # Reading light/mass vectors   
    x = np.zeros(int(nel_base))
    mu_ini = np.zeros(int(nel_base))
    mu_cor = np.zeros(int(nel_base))
    age = np.zeros(int(nel_base))
    for i in range(int(nel_base)):
        x[i] = lines[63+i].split()[1]    
        mu_ini[i] = lines[63+i].split()[2]  
        mu_cor[i] = lines[63+i].split()[3]  
        age[i] = lines[63+i].split()[4]  

    # Normalizing the vectors
    x = x / sum(x)
    mu_ini = mu_ini / sum(mu_ini)
    mu_cor = mu_cor / sum(mu_cor)

    # Calculating the condensed vectors
    x_cond = np.zeros(3)
    mu_ini_cond = np.zeros(3)
    mu_cor_cond = np.zeros(3)

    # Young population (t<1e8 yrs)
    x_cond[0] = sum(x[(age <= 1e8)])
    mu_ini_cond[0] = sum(mu_ini[(age <= 1e8)])
    mu_cor_cond[0] = sum(mu_cor[(age <= 1e8)])

    # Intermediate population (1e8 < t < 1e9 yrs)
    x_cond[1] = sum(x[(age > 1e8) & (age <= 1e9)])
    mu_ini_cond[1] = sum(mu_ini[(age > 1e8) & (age <= 1e9)])
    mu_cor_cond[1] = sum(mu_cor[(age > 1e8) & (age <= 1e9)])

    # Old population (t > 1e9 yrs)
    x_cond[2] = sum(x[(age > 1e9)])
    mu_ini_cond[2] = sum(mu_ini[(age > 1e9)])
    mu_cor_cond[2] = sum(mu_cor[(age > 1e9)])

    return x_cond,mu_ini_cond,mu_cor_cond
#########################################################################
def plot_quartiles(x,y,nbin):
    """
       This routine plots the quartiles of x,y distribution in x bins, 
       showing a possible trend.

                                      Costa-Duarte, M.V. - 05/06/2015
    """
    # Calculate the percentiles limits of the sample in x-axis

    qx = np.zeros(nbin+1)
    delta_bin = 100. / float(nbin)
    for i in range(len(qx)):
        qx[i] = np.percentile(x,delta_bin*i)
        #print qx[i],delta_bin*i
    
    # For each bin, calculate the quartiles and median
    q1_y = np.zeros(len(qx)-1)
    med_y = np.zeros(len(qx)-1)
    q2_y = np.zeros(len(qx)-1)
    x_out = np.zeros(len(qx)-1)
    for i in range(len(qx)-1):

        # Selecting points inside the bin
        idx = [(x >= qx[i]) & (x < qx[i+1])]
        x_out[i] = (qx[i] + qx[i+1]) / 2.
 
        if len(y[idx]) >= 2:

            # Calculating the quartiles and median values inside this x-bin

            q1_y[i] = np.percentile(y[idx],25)
            med_y[i] = np.percentile(y[idx],50)
            q2_y[i] = np.percentile(y[idx],75)

    return q1_y, med_y, q2_y, x_out
########################################################################################
def mass_assembly(infile,Mini,deltat_dex):
    """

     This routine reads the mu_ini and generates a cumulative mass assembly vector based on
     the spectral synthesis results. Basically uses mu_ini and ages of SSPs.

      Following Asari+07 formalism (http://arxiv.org/pdf/0707.3578v1.pdf)

                                       Costa-Duarte, M.V. 26/05/2015
    """

    tmax = 13.5e9 # Maximum age (t < t_age_Universe)
    Zsun = 0.02   # Solar metallicity

    # Read the infile
    lines = [line.strip() for line in open(infile)]
    #
    nel_base = lines[9].split()[0]          # Number of elements of the BASE
    #
    x0 = np.zeros(int(nel_base))
    mu_ini0 = np.zeros(int(nel_base))
    mu_cor0 = np.zeros(int(nel_base))
    age0 = np.zeros(int(nel_base))
    Z0 = np.zeros(int(nel_base))
    for i in range(int(nel_base)):
        x0[i] = lines[63+i].split()[1]      # light-vector
        mu_ini0[i] = lines[63+i].split()[2] # Mu_ini 
        mu_cor0[i] = lines[63+i].split()[3] # Mu_cor 
        age0[i] = lines[63+i].split()[4]    # Age (yrs)
        Z0[i] = lines[63+i].split()[5]      # Metallicity
    #
    # Just in case, normalize the mu_ini
    mu_ini0 = mu_ini0 / sum(mu_ini0)

    # Compiling the mu_ini0 in ages...
    age0_unique = np.unique(age0)
    mu_ini0_unique = np.zeros(len(age0_unique))
    for i in range(len(age0_unique)):
        idx = [(age0 == age0_unique[i])]
        mu_ini0_unique[i] = sum(mu_ini0[idx])
        #print age0_unique[i],mu_ini0_unique[i]

    # Define new AGE array (higher resolution for smoothing)
    age_mass_assembly = np.arange(min(np.log10(age0_unique[(age0_unique <= tmax)])),max(np.log10(age0_unique[(age0_unique <= tmax)])),deltat_dex)

    # Smoothing parameter is the mean differences between the age array elements
    sigma = np.average(abs(age_mass_assembly[:len(age_mass_assembly)-1] - age_mass_assembly[1:]))

    # Smoothed mu_ini (smoothing in log10 scale)
    mu_ini_smoothed = smooth_vector(mu_ini0_unique,np.log10(age0_unique),age_mass_assembly,sigma)

    # Normalizing the mu_ini array again...
    mu_ini_smoothed /= sum(mu_ini_smoothed)

    # Calculate the mass assembly
    mass_assembly = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):
        mass_assembly[i] = sum(mu_ini_smoothed[(age_mass_assembly >= age_mass_assembly[i])])

    # Calculate the SFR(t) and SSFR(t)
    SSFR = np.zeros(len(age_mass_assembly))
    SFR = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):
  
        # SSFR(t) and SFR(t)
        if mu_ini_smoothed[i] > 0.:
            SSFR[i] = np.log10((np.log10(np.exp(1.)) * mu_ini_smoothed[i]) / (10**age_mass_assembly[i] * deltat_dex))
            SFR[i] = np.log10((10**Mini) * (np.log10(np.exp(1.)) * mu_ini_smoothed[i]) / (10**age_mass_assembly[i] * deltat_dex))
        else:
            SSFR[i] = -99999.
            SFR[i] = -99999.
       
    # Calculate Z(t) 
    Z = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):

        # Getting SSPs older than age_mass_assembly[i]
        idx = [(np.log10(age0) >= age_mass_assembly[i])]

        # Define metallicity of the galaxy (weighted by mu_ini)
        Z[i] = sum(mu_cor0[idx] * (Z0[idx] / Zsun)) / sum(mu_cor0[idx])

    return mass_assembly, SFR, SSFR, Z, 10**age_mass_assembly

#########################################################################################

def gaussian_filter(x,mu,sigma):
    """
    This routine draws the gaussian profile.

                                        Costa-Duarte, M.V. 30/05/2015
    """
    return np.exp(-((x-mu)**2)/(2.*sigma**2))

########################################################################################

def smooth_vector(vector,age,new_age,sigma):
    """

    This routine smoothes an array (light/mass vector) from the spectral synthesis
    by using a gaussian filter with "sigma" as standard deviation in log scale (~ 1dex).

                                        Costa-Duarte, M.V. 30/05/2015
    """


    # Define new vector

    new_vector = np.zeros(len(new_age))
 
    for i in range(len(new_vector)):

        # Selecting array points within +- 3sigma
        idx = [(age >= new_age[i] - 3 * sigma) & (age <= new_age[i] + 3 * sigma)]
 
        # Smoothing it...
        new_vector[i] = sum(vector[idx] * gaussian_filter(age[idx],new_age[i],sigma)) / sum(gaussian_filter(age[idx],new_age[i],sigma))  

    return new_vector

########################################################################################

def Z_evolution(infile,deltat_dex):
    """

    This routine reads the population vectors and generates a metallicity evolution
    vector. 
                                        Costa-Duarte, M.V. 20/07/2015

    """
    tmax = 13.5e9 # Maximum age (t < t_age_Universe)
    Zsun = 0.019   # Solar metallicity

    # Read the infile
    lines = [line.strip() for line in open(infile)]
    #
    nel_base = lines[9].split()[0]         # Number of elements of the BASE
    #
    x0 = np.zeros(int(nel_base))
    mu_ini0 = np.zeros(int(nel_base))
    mu_cor0 = np.zeros(int(nel_base))
    age0 = np.zeros(int(nel_base))
    Z0 = np.zeros(int(nel_base))
    for i in range(int(nel_base)):
        x0[i] = lines[63+i].split()[1]      # light-vector
        mu_ini0[i] = lines[63+i].split()[2] # Mu_ini 
        mu_cor0[i] = lines[63+i].split()[3] # Mu_cor 
        age0[i] = lines[63+i].split()[4]    # Age (yrs)
        Z0[i] = lines[63+i].split()[5]      # Metallicity
    #
    # Just in case, normalize the mu_ini
    mu_ini0 = mu_ini0 / sum(mu_ini0)

    # Define new AGE array (higher resolution for smoothing)
    age_mass_assembly = np.arange(min(np.log10(age0[(age0 <= tmax)])),max(np.log10(age0[(age0 <= tmax)])),deltat_dex)

    # Smoothing parameter is the mean differences between the age array elements
    sigma = np.average(abs(age_mass_assembly[:len(age_mass_assembly)-1] - age_mass_assembly[1:]))

    # Smoothed mu_init (smoothing in log10 scale)
    mu_ini_smoothed = smooth_vector(mu_ini0,np.log10(age0),age_mass_assembly,sigma)

    # Normalizing the mu_ini array again...
    mu_ini_smoothed /= sum(mu_ini_smoothed)
    #
    # Calculate the mass assembly
    mass_assembly = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):
        mass_assembly[i] = sum(mu_ini_smoothed[(age_mass_assembly >= age_mass_assembly[i])])

    # Calculate the SFR(t) and SSFR(t)
    SSFR = np.zeros(len(age_mass_assembly))
    SFR = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):
  
        # SSFR(t) and SFR(t)
        if mu_ini_smoothed[i] > 0.:
            SSFR[i] = np.log10((np.log10(np.exp(1.)) * mu_ini_smoothed[i]) / (10**age_mass_assembly[i] * deltat_dex))
            SFR[i] = np.log10((10**Mini) * (np.log10(np.exp(1.)) * mu_ini_smoothed[i]) / (10**age_mass_assembly[i] * deltat_dex))
        else:
            SSFR[i] = -99999.
            SFR[i] = -99999.
       
    # Calculate Z(t) 
    Z = np.zeros(len(age_mass_assembly))
    for i in range(len(age_mass_assembly)):

        # Metallicity weighted by mass-vector
        idx = [(age0 <= 10 ** age_mass_assembly[i])]
        Z[i] = sum(mu_cor0[idx] * (Z0[idx] / Zsun)) / sum(mu_cor0[idx]) 

        # Metallicity itself (from current stellar population
        #idx = [(abs(age_mass_assembly_gal[i] - age0) == min(abs(age_mass_assembly_gal[i] - age0)))]
        #Z[i] = Z0[idx] / Zsun)
    #print 'bla - mass assembly'

    return mass_assembly, SFR, SSFR, Z, 10**age_mass_assembly

#############################################################################

def convert_ra_dec_xyz(alpha,delta,dist,alpha0,dec0):
    """

    Convert ra,dec and comoving distance to cartesian coordinates (x,y,z)
    It is also necessary to give a "shift" in ra to align the cone (alpha0).
    ATTENTION: ra and dec in degrees!! 
                                            13/10/2015 - Costa-Duarte, M.V. 

    """

    # Rotating the data in order to get in vertical alignment

    if alpha0 == -99999.:
        alpha0 = np.average(alpha) - 90.
    if dec0 == -99999.:
        delta0 = np.average(delta)

    # ra,dec -> x,y,z

    conv = np.pi / 180.
    x = dist * np.cos((delta - delta0) * conv) * np.cos((alpha - alpha0) * conv)
    y = dist * np.cos((delta - delta0) * conv) * np.sin((alpha - alpha0) * conv)
    z = dist * np.sin((delta - delta0) * conv)

    return x,y,z

##################################################################

def bootstrap(x,n_realization):
    """
        This routine estimates the error of a certain set of 
        measures (x), estimating the errors over n_realizations.
        
        In this case the final error is the standard deviation of
        the averages over n_realizations (subsamples of frac_subsample).
        
                                     28/10/2015 - Marcus

     Bootstrap with medians: https://onlinecourses.science.psu.edu/stat464/node/80
 
    """    
    if len(x) == 1: 
        #print 'len(x) == 1 -> error = 0!'
        return 0.
    if len(x) == 0:
        #print 'array with no length!'
        exit()
    #
    x0 = np.zeros(n_realization)
    for i in range(n_realization):
        # samples with replacement and with size of the initial sample
        x_sample = np.random.choice(x,len(x)) 
        x0[i] =  np.average(x_sample) 
    return np.std(x0)

######################################################################

def rescale_array(x, xmin_new, xmax_new):
    """
        This routine "standardlize" a vector x, being the new vector
        a linear scale between new range [xmin_new, xmax_new].

                                            Marcus - 04/04/2016
    """
    if max(x) == min(x): 
        print '[rescale_array] zero range of array!!'
        exit()
 
    return xmin_new + (xmax_new - xmin_new) * (x - min(x)) / (max(x) - min(x))

######################################################################

def load_mag_error(path_tables, infile_mean, infile_sigma, nbands):

    """
        This routine loads the J-PLUS magnitude uncertainties
        and gives it back as matrices

                                        Marcus - 17/05/2016
    """

    # Read the infile_mag

    lines = [line.strip() for line in open(path_tables + infile_mean)]
    lines = lines[1:]

    mag_range_mean = np.zeros(len(lines))
    mag_mean = np.zeros(len(lines) * nbands).reshape(len(lines), nbands)
    #
    nline = -1
    for line in lines:

        p = line.split()

        if p[0] <> '#':

            nline += 1
            mag_range_mean[nline] = float(p[0])

            for i in range(nbands):

                mag_mean[nline, i] = float(p[1 + i])

    # Read the infile_sigma

    lines = [line.strip() for line in open(path_tables + infile_sigma)]
    lines = lines[1:]

    mag_range_sigma = np.zeros(len(lines))
    mag_sigma = np.zeros(len(lines) * nbands).reshape(len(lines), nbands)
    #
    nline = -1
    for line in lines:

        p = line.split()

        if p[0] <> '#':

            nline += 1
            mag_range_sigma[nline] = float(p[0])

            if mag_range_sigma[nline] <> mag_range_mean[nline]:
                print 'mag arrays does not match!! [load_mag_error]'
                exit()

            for i in range(nbands):

                mag_sigma[nline, i] = float(p[1 + i])
   
    return mag_range_mean, mag_mean, mag_sigma

######################################################################

def mag_uncertainty(mag_range, mag_mean, mag_sigma, mag, idx_mag, lowest_mag_uncertainty):

    """
        This routine interpolates the matrices from J-PLUS magnitude
        uncertainties and gives you an uncertainty, given the band index
        and the magnitude value. 

                                        Marcus - 17/05/2016
    """
    mag_mean_out = -99999.
    mag_sigma_out = -99999. 

    # Magnitudes brighter than 14, proabably are saturated ones.
    
    #if mag < 14:
    #    print 'magnitude lower than 14! Probably saturated [mag_uncertainty]'
    #    exit()

    # Defining the magnitude bin which the "mag" is contained

    idx_mag_range = -99999.
    for i in range(len(mag_range)-1):
        if mag >= mag_range[i] and mag <= mag_range[i+1]:
            idx_mag_range = i

    if mag < mag_range[0]:
         #print 'extrapolating magnitude uncertainty to lower limit [mag_uncertainty]',mag_range[0], mag
         mag_mean_out = mag_mean[0, idx_mag]
         mag_sigma_out = lowest_mag_uncertainty # lowest we can consider for a magnitude.

    if mag > mag_range[len(mag_range)-1]: 
         #print 'extrapolating magnitude uncertainty to higher limit [mag_uncertainty]', mag_range[len(mag_range)-1], mag
         idx_mag_range = len(mag_range)-2

    if mag_mean_out == -99999. and mag_sigma_out == -99999.:
        #print 'mag=',mag
        #print 'idx_mag=',idx_mag
        #print 'idx_mag_range=',idx_mag_range
        #print np.shape(mag_range)
        #print np.shape(mag_mean)

        #print mag_range[idx_mag_range], mag_range[idx_mag_range + 1]
        #print mag_mean[idx_mag_range, idx_mag], mag_mean[idx_mag_range + 1, idx_mag]
    
        mag_mean_out = np.interp(mag, [mag_range[idx_mag_range], mag_range[idx_mag_range + 1]],\
        [mag_mean[idx_mag_range, idx_mag], mag_mean[idx_mag_range + 1, idx_mag]])
    
        mag_sigma_out = np.interp(mag, [mag_range[idx_mag_range], mag_range[idx_mag_range + 1]],\
        [mag_sigma[idx_mag_range, idx_mag], mag_sigma[idx_mag_range + 1, idx_mag]])

    return mag_mean_out, mag_sigma_out

######################################################################

def read_NGSL_spectra(infile, path_spectra, l_out, l_norm_i, l_norm_f):

    """
        This routine reads an input ASCII table and gives back 
        a 2D-list with data. 

                                        Marcus - 18/05/2016
    """

    # Reading list of spectra

    f = open(infile,'r')
    data = f.readlines()
    f.close()

    file_spec_star = []
    for line in data:
        p = line.split()
        if p[0] <> '#':
            file_spec_star.append(str(p[0]))

    # Loading list of spectra

    star_spectra = np.zeros(len(file_spec_star) * len(l_out)).reshape(len(file_spec_star), len(l_out))
    fobsnorm_stars = np.zeros(len(file_spec_star))
    for i in range(len(file_spec_star)):

        #print i, file_spec_star[i],len(file_spec_star)
        f = open(path_spectra + file_spec_star[i], 'r')
        data = f.readlines()
        f.close()

        l = []
        f = []
        for line in data:
            p = line.split()
            if p[0] <> '#':
                l.append(float(p[0]))
                f.append(float(p[1]))
        f = np.array(f)
        l = np.array(l)

        # Interpolate star spectrum to be in the same lambda array

        star_spectra[i] = np.interp(l_out, l, f)

        # Normalization factor of the spectrum

        fobsnorm_stars[i] = norm_spec(l_out, star_spectra[i],  l_norm_i, l_norm_f)

    return star_spectra, fobsnorm_stars

#############################################################

def calculate_leff(l, R):
    """
     Calculate lambda_eff of a filter transmission

                                           Costa-Duarte - 18/05/2016
    """

    if len(l) <> len(R):
        print '[leff starlight_tools] len(l) <> l(R)!'
        exit()

    leff = 0.
    sum_lRdl = 0.
    sum_ldl = 0.

    for i in range(len(l)-1): # Trapezoidal numerical integration

        sum_lRdl += 0.5 * l[i] * (R[i+1] + R[i]) * (l[i+1] - l[i])
        sum_ldl += 0.5 * (R[i+1] + R[i]) * (l[i+1] - l[i])

    leff = sum_lRdl / sum_ldl    

    return leff

########################################################################################

def read_synthesis_v5(infile):
    """
     This routine reads spectral synthesis results and calculate ages and metallicities (STARLIGHTv5).

                                                  Costa-Duarte, M.V. - 18/05/2016
    """

#    Read the infile
    lines = [line.strip() for line in open(infile)]
#
    flag_synthesis_ok = 0
    if len(lines) > 3: # Some error and STARLIGHT could not fit it...
        nel_base = lines[9].split()[0]                                            # Number of elements of the BASE
        SN = float(lines[30].split()[0])                                           # sign-to-noise ratio
        sum_x = float(lines[52].split()[0])                                        # Sum of the light vector 
        Mini = np.log10(float(lines[54].split()[0]))                           # Initial stellar mass
        Mcor = np.log10(float(lines[55].split()[0]))                           # Corrected stellar mass 
        v0 = float(lines[57].split()[0])                                           # v0
        sigma = float(lines[58].split()[0])                                        # velocity dispersion
        AV =  float(lines[59].split()[0])                                          # Extinction parameter
        Nl =  float(lines[38].split()[0])                                          # # of lambdas (initial)
        Nl_clipped =  float(lines[40].split()[0])                                  # # of lambdas clipped
        frac_clipped = Nl_clipped / Nl
        chi2_red = float(lines[49].split()[0])                                     # reduced Chi2

        # Reading light/mass vectors   
        x = np.zeros(int(nel_base))
        mu = np.zeros(int(nel_base))
        age = np.zeros(int(nel_base))
        Z = np.zeros(int(nel_base))
        for i in range(int(nel_base)):
            x[i] = lines[63+i].split()[1]    
            mu[i] = lines[63+i].split()[3]  
            age[i] = lines[63+i].split()[4]  
            Z[i] = lines[63+i].split()[5]
        #    print i+1,x[i],mu[i],str(age[i]),np.log10(age[i])
        #
        # Calculate average ages/metallicities 
        agelight = np.dot(np.log10(age),x)/sum(x)   
        agemass = np.dot(np.log10(age),mu)/sum(mu)   
        zlight = np.dot(np.log10(Z),x)/sum(x)    
        zmass = np.dot(np.log10(Z),mu)/sum(mu)
    else:
        flag_synthesis_ok = -99999.
        nel_base = -99999.
        SN = -99999.
        Mcor = -99999.
        Mini = -99999.
        AV = -99999.
        agemass = -99999.
        agelight = -99999.
        zmass = -99999.
        zlight = -99999.
        v0 = -99999.
        sigma = -99999. 
    return nel_base, SN, Mcor, Mini, AV, agemass, agelight, zmass, zlight, v0, sigma, frac_clipped, chi2_red, flag_synthesis_ok

########################################################################################

def read_ascii_table(infile, header_string, columns, type_variable):
    """

     This routine reads an ASCII table, taking specific columns only. 
     INPUT: index of columns + the type of vatiables.

                                                  Costa-Duarte, M.V. - 18/05/2016 
    """

    columns = np.array(columns)
    type_variable = np.array(type_variable)

    # Arrays columns and type_variables have different lengths

    if len(columns) <> len(type_variable): 
        print '[read_ascii_table] len(columns) <> len(type_variables)'
        exit()

    # Load table
    f = open(infile,'r')
    data = f.readlines()
    f.close()

    matrix = []
    for i in range(len(columns)):
        matrix.append([])

    n0 = -1
    for line in data:
        p = line.split()
        nline = -1
        n0 += 1
        #print n0, p[1]
        if p[0] <> header_string: 
            for i in range(len(columns)):
                nline += 1
                if type_variable[i] == 0: # string
                    matrix[nline].append(str(p[columns[i]]))
                if type_variable[i] == 1: # float
                    matrix[nline].append(float(p[columns[i]]))
                if type_variable[i] == 2: # integer
                    matrix[nline].append(int(p[columns[i]]))
                if type_variable[i] > 2 or type_variable[i] < 0:
                    print '[read_ascii_table] type_variables -> bad values!!'
                    exit()
    return matrix

########################################################################################

def read_eline_BELFI(infile):

    """
     This routine reads the output of the Bayesian Emission Line Fitting Code (BELFI). 

                                                  Costa-Duarte, M.V. - 18/05/2016

    """

    # Load table
    f = open(infile,'r')
    data = f.readlines()
    f.close()

    l0 = []
    F = []
    dF = []
    EW = []
    dEW = []
    SN = []

    for line in data:
        p = line.split()
        if p[0] <> 'l0': 
           l0.append(float(p[1]))
           F.append(float(p[2]))
           dF.append(float(p[3]))
           EW.append(float(p[4]))
           dEW.append(float(p[5]))
           SN.append(float(p[12]))

    l0 = np.array(l0)
    F = np.array(F)
    dF = np.array(dF)
    EW = np.array(EW)
    dEW = np.array(dEW)

    return l0, F, dF, EW, dEW