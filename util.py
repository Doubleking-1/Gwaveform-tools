from pycbc import conversions

def get_parameter(mtotal,nr_metadata,distance=100,eccentricity=0,inclination=0,coa_phase=0):
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
    q = float(nr_metadata['relaxed-mass-ratio-1-over-2'])
    if q<1:
        q = 1/q
        
    kwargs = {'mass1':conversions.mass1_from_mtotal_q(mtotal,q),
          'mass2':conversions.mass2_from_mtotal_q(mtotal,q),
          'spin1z':float(nr_metadata['relaxed-chi1z']),
          'spin2z':float(nr_metadata['relaxed-chi2z']),
          'distance':distance,
          'inclination':inclination,
          'coa_phase':coa_phase,
          'eccentricity':eccentricity
             }
    return kwargs