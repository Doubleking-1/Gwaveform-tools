from pycbc import conversions

def get_parameter(mtotal,nr_metadata,distance=100,eccentricity=0,inclination=0,coa_phase=0,
                 catalog='rit'):
    '''
    Given a mtotal and nr_metadata, return waveform par to generate pycbc waveform
    or convert a NR waveform to phy unit
        
    Parameters
    -----------
    mtotal: float
        The total mass of the binary
    nr_metadata: dict
        Metadata of a RIT waveform
    distance: float
        Luminosity distance. Default: 100
    eccentricity: float
        Eccentricity. Default: 0.
    inclination: float
        Inclination angle. Default: 0.
    coa_phase: float
        Coalescence phase. Default: 0
    
    Return
    -----------
    kwargs: dict
        Waveform parameter dictionary
    '''
    #mass ratio from NR metadata
    if catalog == 'rit':
        q = float(nr_metadata['relaxed-mass-ratio-1-over-2'])
        spin1z = float(nr_metadata['relaxed-chi1z'])
        spin2z = float(nr_metadata['relaxed-chi2z'])
    elif catalog == 'sxs':
        q = float(nr_metadata['reference_mass_ratio'])
        spin1z = float(nr_metadata['reference_chi1_perp'])
        spin2z = float(nr_metadata['reference_chi2_perp'])
    else:
        raise ValueError
        
    if q<1:
        q = 1/q
        
    kwargs = {'mass1':conversions.mass1_from_mtotal_q(mtotal,q),
          'mass2':conversions.mass2_from_mtotal_q(mtotal,q),
          'spin1z':spin1z,
          'spin2z':spin2z,
          'distance':distance,
          'inclination':inclination,
          'coa_phase':coa_phase,
          'eccentricity':eccentricity
             }
    return kwargs