from pycbc.types import TimeSeries
import lal,sxs

class sxswave(object):
    '''
    A class to read SXS waveform
    At the moment only return h22 and sliced waveform starting from the reference time
    '''
    def __init__(self,sxsid,ext_order=4):
        #load the data and metadata.json
        self.hlm = sxs.load("SXS:BBH:"+str(sxsid)+"/Lev/rhOverM", extrapolation_order=ext_order)
        self.metadata = sxs.load("SXS:BBH:"+str(sxsid)+"/Lev/metadata.json")

        # time slide the 0 time point to the maximum norm position
        # self.max_norm_time = self.hlm.max_norm_time()
        # AttributeError: can't set attribute
        
        # chop off the junk radiation
        t_start = self.metadata.reference_time
        i_start = self.hlm.index_closest_to(t_start)
        self.hlm_sliced = self.hlm[i_start:]
        
        # Only consider 22 mode at the moment
        self.h22 = self.hlm_sliced[:, self.hlm.index(2, 2)]
        
    def hphc(self,theta,phi):
        '''
        Evaluate h+ - ihx = hlm -2Ylm(theta,phi)
        '''
        return self.hlm_sliced.evaluate(theta, phi)
    
    def h22_pycbcts(self):
        '''
        h22
        '''
        t_start = self.h22.t[0]
        t_end = self.h22.t[-1]
        dt = np.max(np.diff(self.h22.t))
        #uniform time to interpolate
        t_uniform = np.arange(t_start, t_end, dt)
        h = self.h22.interpolate(t_uniform)
        return TimeSeries(h.data, delta_t = dt,
                          epoch = -dt * np.argmax(np.abs(h.data)))
    
    def hp_m_ihc_phyunit(self):
        '''
        h22
        '''
        t_start = self.h22.t[0]
        t_end = self.h22.t[-1]
        dt = np.max(np.diff(self.h22.t))
        #uniform time to interpolate
        t_uniform = np.arange(t_start, t_end, dt)
        h = self.h22.interpolate(t_uniform)
        return TimeSeries(h.data, delta_t = dt,
                          epoch = -dt * np.argmax(np.abs(h.data)))
    
    def hp_pycbcts_phyunit(self,**kwargs):
        '''
        
        '''
        mtotal = kwargs['mass1'] + kwargs['mass2']
        distance = kwargs['distance']
        amp_factor = distance*1e6*lal.PC_SI / mtotal /lal.MRSUN_SI
        time_factor = 1/ mtotal /lal.MTSUN_SI

        '''
        if dt / time_factor < kwargs['delta_t']:
            dt  = time_factor * kwargs['delta_t']
        else:
            raise ValueError("dt (in physics unit) larger than the delta_t required")
        '''
        h22 = self.h22_pycbcts()
        Y22 = lal.SpinWeightedSphericalHarmonic(0,0,-2,2,2)
        hplus = np.real( h22.data * Y22)
        dt = h22.delta_t
        return TimeSeries( hplus / amp_factor,delta_t = dt / time_factor,\
                          epoch = - dt * np.argmax(np.abs(hplus)) / time_factor)