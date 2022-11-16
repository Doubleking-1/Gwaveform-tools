import lal,romspline
from pycbc.types import TimeSeries
import glob,os,json
import numpy as np

class RITwave(object):
    '''
    Load in a numerical relativity waveform from RIT catalog given a waveform id.
    At the moment only return h22.
    '''
    def __init__(self,waveid,modes=[[2,2]],nrpath='/work/francisco.jimenez/RIT/'):
        '''
        Parameters
        -----------
        waveid: int
            The ID number of a RIT waveform
        modes: list of list
            A list of harmonic modes to be loaded
        nrpath: str
            The prefix of the directory to store RIT waveforms
        '''
        self.h5file_path = glob.glob(nrpath+'/Data/ExtrapStrain_RIT-eBBH-'+str(waveid)+'-n*.h5')[0]
        self.metadata_path = glob.glob(nrpath+'/Metadata/RIT:eBBH:'+str(waveid)+'-n*-ecc_Metadata.json')[0]
        
        if os.path.exists(self.h5file_path) and os.path.exists(self.metadata_path):
            f = open(self.metadata_path)
            self.metadata = json.load(f)
        else:
            raise FileNotFoundError
   
        self.modes = modes
        self.id = waveid
        
    def get_single_mode(self,mode):
        '''
        Read in h_lm given a pair of (l,m).
        
        Parameters
        -----------
        mode: list
            A list of harmonic modes to be loaded, e.g., [2,2]
            
        Return
        -----------
        h: pycbc.types.TimeSeries
            A time domain series of h_lm.
        '''
        fp = self.h5file_path
        get_amp=romspline.readSpline(fp,group='amp_l'+str(mode[0])+'_m'+str(mode[1]))
        get_phase=romspline.readSpline(fp,group='phase_l'+str(mode[0])+'_m'+str(mode[1]))

        time_amp=get_amp.X
        amp = get_amp.Y
        time_phase=get_phase.X
        phase=get_phase.Y
                
        #Interpolation function using Reduced Order Spline
        amp_int = romspline.ReducedOrderSpline(time_amp, amp,verbose=False)
        phase_int = romspline.ReducedOrderSpline(time_phase, phase,verbose=False)
        
        #Construct a uniform time array
        tmin=max(time_phase[0],time_amp[0])
        tmax=min(time_phase[-1],time_amp[-1])
        dt = max(min(np.diff(time_phase)),min(np.diff(time_amp)))
        #TODO: control the resolution to be min(dt)=1
        if dt<1:
            dt=1.0
        #TODO: how to choose dt
        times=np.arange(tmin,tmax,dt)
        
        #Construct h_lm = amp * exp(-i*phase)
        h=amp_int(times)*np.exp(-1j*phase_int(times)) 
        return TimeSeries(h, delta_t = dt,epoch = -dt * np.argmax(np.abs(h)))
    
    def h22(self):
        '''
        h_22

        Return
        -----------
        h22: pycbc.types.TimeSeries
            A time domain series of h_22.
        '''
        h = self.get_single_mode([2,2])
        return h
    
    def hp_m_ihc_phyunit(self,**kwargs):
        '''
        h_+-ih_x in physical unit
        
        Parameters
        -----------
        kwargs: dict
            A dictionary of waveform parameter, including mass1,
            mass2,distance,inclination,coa_phase
            
        Return
        -----------
        hplus: pycbc.types.TimeSeries
            A time domain series of h_+ from (2,2) mode.
        '''
        mtotal = kwargs['mass1'] + kwargs['mass2']
        distance = kwargs['distance']
        inclination = kwargs['inclination']
        coa_phase = kwargs['coa_phase']
        
        amp_factor = distance*1e6*lal.PC_SI / mtotal /lal.MRSUN_SI
        time_factor = 1/ mtotal /lal.MTSUN_SI
        
        h22 = self.h22()
        Y_22 = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, l=2, m=2)
        #h_+ - ih_x = h_22 * -2Y_22
        hp_m_ihc = h22.data * Y_22
        dt = h22.delta_t
        return TimeSeries( hp_m_ihc / amp_factor,delta_t = dt / time_factor,\
                          epoch = - dt * np.argmax(np.abs(hp_m_ihc)) / time_factor)
    
    def hp22_phyunit(self,**kwargs):
        '''
        h_plus in physical unit
        
        Parameters
        -----------
        kwargs: dict
            A dictionary of waveform parameter, including mass1,
            mass2,distance,inclination,coa_phase
            
        Return
        -----------
        hplus: pycbc.types.TimeSeries
            A time domain series of h_+ from (2,2) mode.
        '''
        mtotal = kwargs['mass1'] + kwargs['mass2']
        distance = kwargs['distance']
        inclination = kwargs['inclination']
        coa_phase = kwargs['coa_phase']
        
        amp_factor = distance*1e6*lal.PC_SI / mtotal /lal.MRSUN_SI
        time_factor = 1/ mtotal /lal.MTSUN_SI
        
        h22 = self.h22()
        Y_22 = lal.SpinWeightedSphericalHarmonic(inclination, coa_phase, -2, l=2, m=2)
        #h_+ - ih_x = h_22 * -2Y_22
        hplus = np.real( h22.data * Y_22)
        dt = h22.delta_t
        return TimeSeries( hplus / amp_factor,delta_t = dt / time_factor,\
                          epoch = - dt * np.argmax(np.abs(hplus)) / time_factor)