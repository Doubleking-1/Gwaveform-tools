from scipy.optimize import minimize_scalar

from pycbc import conversions
from pycbc.waveform import get_td_waveform,td_taper,apply_fd_time_shift
from pycbc.psd import aLIGOZeroDetHighPower
from pycbc.filter import match,matched_filter

import lal,json,os,glob,numpy as np
from tqdm import tqdm

from .utils import get_parameter
from .loadrit import RITwave

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.use('Agg')
# PLOTTING OPTIONS
fig_width_pt = 3*246.0  # Get this from LaTeX using \showthe\columnwidth
inches_per_pt = 1.0/72.27               # Convert pt to inch
golden_mean = (np.sqrt(5)-1.0)/2.0         # Aesthetic ratio
fig_width = fig_width_pt*inches_per_pt  # width in inches
fig_height = fig_width*golden_mean      # height in inches
fig_size =  [fig_width,fig_height]

params = { 'axes.labelsize': 24,
          'font.family': 'serif',
          'font.serif': 'Computer Modern Raman',
          'font.size': 24,
          'legend.fontsize': 20,
          'xtick.labelsize': 24,
          'ytick.labelsize': 24,
          'axes.grid' : True,
          'text.usetex': True,
          'savefig.dpi' : 100,
          'lines.markersize' : 14,
          'figure.figsize': fig_size}

mpl.rcParams.update(params)

def get_psd(seob,nr,start_frequency=15):
    '''
    Generate PSD using LIGO designed sensitivity.
    
    Parameters
    -----------
    seob: pycbc.type.TimeSeries
        hplus from SEOBNREHM waveform
    nr: pycbc.type.TimeSeries
        hplus from numerical relativity
    start_frequency: float
        low frequency cutoff for PSD generation. Default: 15
    
    Return
    -----------
    psd: pycbc.types.FrequencySeries
        Noise PSD of LIGO designed sensitivity
    '''
    if seob.duration != nr.duration:
        raise TypeError('Two waveforms have unequal duration!')

    tlen = len(seob)
    flen = tlen//2 + 1
    delta_f = 1.0 / seob.duration
    psd = aLIGOZeroDetHighPower(flen, delta_f, start_frequency)
    return psd

def get_overlap(eccentricity,mtotal,nr_class,taper_fraction=0.1,flow_seobnre=20,validation=False):
    '''
    Get overlap between a SEOBNREHM waveform and a numerical relativity waveform.
    Overlap is the maximized match between two waveforms over the coalescence time
    and phase. It's a function of total mass, also of eccentricity in this research
    context. It tapers the start of the wavefrom using 0.1 s. If validation is false,
    it only returns the overlap. If validation is true, return an overlap and the 
    associated phase being maximized over. It can be used to shift the waveform to be 
    aglined.
    
    Parameters
    -----------
    eccentricity: float
        eccentricity of SEOBNREHM waveform
    mtotal: float
        total mass
    nr_class: instance
        instance of the RITwave class
    taper_fraction: float
        The fraction of length of taper window. default: 0.1 (10%)
    flow_seobnre: float
        starting frequency of SEOBNREHM. default: 18 Hz
    validation: bool
        if true, return coalescence phase which maximize the overlap
        
    Return
    -----------
    m: float
        overlap between SEOBNREHM and a numerical relativity waveform
    or
    hp_shift,nr_taper: TimeSeries
        the shifted SEOBNREHM waveform and a numerical relativity waveform
    '''
    par = get_parameter(mtotal,nr_class.metadata)
    #TODO: the parameter may also depend on orientation angles.
    par.update({'eccentricity':eccentricity})

    #Use pycbc.waveform.utils.td_taper to taper the waveform
    nr_phy = nr_class.hp22_phyunit(**par)
    taper_window = nr_phy.duration * taper_fraction
    nr_taper = td_taper(nr_phy,nr_phy.start_time,nr_phy.start_time+taper_window)

    dt = nr_taper.delta_t
    nr_flow22 =  float(nr_class.metadata['freq-start-22-Hz-1Msun'])/mtotal
    
    if nr_flow22 > 1000:
        raise ValueError('NR flow22 is greater than 1000 Hz!')
    #start from either flow_seobnre Hz, lower than that noise is dominated, or the 
    #starting frequency of the numerical relativity waveform
    #TODO: should I fix the starting frequency so that the comparison is fair?
    match_flow = max(flow_seobnre,nr_flow22)
    
    #if condition to check mtotal is not too big
    fini_highest = 10.5**(-1.5)/np.pi/lal.MTSUN_SI/mtotal
    if fini_highest <= match_flow:
        raise ValueError('The total mass is too high. The highest allowed initial frequency'
                         'of SEOBNREHM is %.2f,while the initial frequency is %.2f' % 
                         (fini_highest,match_flow))
    
    #Generate SEOBNREHM waveform
    hp, _ = get_td_waveform(**par,
                                 approximant='SEOBNREHM',
                                 delta_t=dt,
                                 f_lower=match_flow,is_only_22=1)
    taper_window = hp.duration * taper_fraction
    seob_taper = td_taper(hp,hp.start_time,hp.start_time+taper_window)
    
    #resize to align waveforms
    tlen = max(len(seob_taper),len(nr_taper))
    seob_taper.resize(tlen)
    nr_taper.resize(tlen)
    
    #get PSD
    psd = get_psd(seob_taper,nr_taper)
    
    #get match
    if not validation:
        m, _ = match(seob_taper, nr_taper, psd=psd, low_frequency_cutoff=match_flow)
        return m
    else:
        mf = matched_filter(seob_taper, nr_taper, psd=psd, low_frequency_cutoff=match_flow)
        idx_max   = np.argmax(np.abs(mf))
        max_time  = mf.sample_times[idx_max] 
        max_phase = np.angle(mf[idx_max])
        hp_shift = seob_taper.copy()
        hp_shift = hp_shift.to_frequencyseries() * np.exp(1.0j*max_phase)
        hp_shift = apply_fd_time_shift(hp_shift, max_time, copy=True)
        hp_shift = hp_shift.to_timeseries()
        return hp_shift,nr_taper
    
def max_overlap_over_ecc(mtotal,nr_class,ecc_upper=0.4):
    '''
    Find the eccentricity to maximize overlap
    
    Parameters
    -----------
    mtotal: float
        total mass
    nr_class: instance
        an instance of the RITwave class
        
    Return
    -----------
    ecc: float
        eccentricity that maximize the overlap
    overlap: float
        overlap between SEOBNREHM and a numerical relativity waveform
    '''
    
    #target funciton to minimize
    def _target(e):
        return -get_overlap(e,mtotal,nr_class)
    
    #minimize the target using scipy.optimize.minimize_scalar
    try:
        res = minimize_scalar(_target, bounds=(0, ecc_upper), method='bounded')
    except ValueError:
        return np.nan,np.nan
    
    #get the results
    ecc = res.x
    overlap = -1*_target(res.x)
    return ecc,overlap

def plot_overlap_vs_mtotal(mtotal,overlap,ecc,nr,output_prefix):
    '''
    
    Parameters
    ----------
    mtotal: numpy.array
    
    overlap: numpy.array
    
    output_prefix: str
    '''
    fig = plt.figure(figsize=([16,16]))
    ax = fig.add_subplot(221)
    ax.plot(mtotal,overlap,'o-')
    ax.set_xlabel('Total mass / $M_\odot$')
    ax.set_ylabel('Overlap')
    
    q = float(nr.metadata['relaxed-mass-ratio-1-over-2'])
    if q<1:
        q = 1/q
    nr_e = float(nr.metadata['eccentricity'])
    chi1z = float(nr.metadata['initial-bh-chi1z'])
    chi2z = float(nr.metadata['initial-bh-chi2z'])
    
    ii = np.nanargmax(overlap)
    seob_valid,nr_valid = get_overlap(ecc[ii],mtotal[ii],nr,taper_fraction=0.1,validation=True)
    bx = fig.add_subplot(222)
    bx.plot(seob_valid.sample_times,seob_valid,label='SEOBNREHM-shift')
    bx.plot(nr_valid.sample_times,nr_valid,label='RIT:'+str(nr.id))
    bx.set_xlabel('Time / s')
    bx.set_ylabel('Strain')
    bx.legend()
    bx.set_title('Overlap:{%.4f}' % overlap[ii]+', mtotal:' +str(mtotal[ii])+', q:{%.2f}' % q+', chi1z:{%.1f}' % chi1z +', chi2z:{%.1f}' % chi2z+
                 '\n NRecc:{%.4f}' % nr_e +', eobecc:{%.4f}' % ecc[ii],fontsize=16)
    
    cx = fig.add_subplot(223)
    cx.plot(seob_valid.sample_times,seob_valid,label='SEOBNREHM-shift')
    cx.plot(nr_valid.sample_times,nr_valid,label='RIT:'+str(nr.id))
    cx.set_xlabel('Time / s')
    cx.set_ylabel('Strain')
    cx.set_xlim(-0.1,0.05)
    cx.legend()
    cx.set_title('Zoom in',fontsize=20)
    
    dx = fig.add_subplot(224)
    seob_f = seob_valid.to_frequencyseries()
    nr_f = nr_valid.to_frequencyseries()
    dx.loglog(seob_f.sample_frequencies,np.abs(seob_f),label='SEOBNREHM')
    dx.loglog(nr_f.sample_frequencies,np.abs(nr_f),label='RIT:'+str(nr.id))
    dx.axvline(float(nr.metadata['freq-start-22-Hz-1Msun'])/mtotal[ii],ls='--',color='gray',label='NR fstart 22')
    dx.legend()
    dx.set_xlabel('Frequency / Hz')
    dx.set_ylabel('$|h_+|$',fontsize=18)
    
    #create output folder
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
        
    fig.savefig(output_prefix+'/RIT-'+str(nr.id)+'.pdf',bbox_inches='tight')
    
def max_overlap_over_mtotal(nrid,mtotal = np.arange(20,200,5),
                         output_prefix='/work/yifan.wang/eccentricity/gitlab-summer-internship/results/'):
    ecc_list = []
    overlap_list = []
    
    nr = RITwave(nrid)
    
    if isinstance(mtotal,int) or isinstance(mtotal,float):
        mtotal = [mtotal]
    
    for m in tqdm(mtotal):
        ecc,overlap = max_overlap_over_ecc(m,nr)
        ecc_list.append(ecc)
        overlap_list.append(overlap)
        
    #create output folder
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
    
    output_fn = output_prefix + '/overlap-RITid-'+str(nr.id)+'.txt'
    np.savetxt(output_fn,np.transpose([mtotal,ecc_list,overlap_list]),
               fmt='%.1f %.8f %.8f',header='mtotal ecc overlap')
    plot_overlap_vs_mtotal(mtotal,overlap_list,ecc_list,nr,
                       output_prefix + '/fig/')