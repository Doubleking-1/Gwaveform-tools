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

def plot_overlap_vs_mtotal(nrid,mtotal,overlap,ecc,nr,output_prefix):
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
    bx.plot(nr_valid.sample_times,nr_valid,label='RIT:'+str(nrid))
    bx.set_xlabel('Time / s')
    bx.set_ylabel('Strain')
    bx.legend()
    bx.set_title('Overlap:{%.4f}' % overlap[ii]+', mtotal:' +str(mtotal[ii])+', q:{%.2f}' % q+', chi1z:{%.1f}' % chi1z +', chi2z:{%.1f}' % chi2z+
                 '\n NRecc:{%.4f}' % nr_e +', eobecc:{%.4f}' % ecc[ii],fontsize=16)
    
    cx = fig.add_subplot(223)
    cx.plot(seob_valid.sample_times,seob_valid,label='SEOBNREHM-shift')
    cx.plot(nr_valid.sample_times,nr_valid,label='RIT:'+str(nrid))
    cx.set_xlabel('Time / s')
    cx.set_ylabel('Strain')
    cx.set_xlim(-0.1,0.05)
    cx.legend()
    cx.set_title('Zoom in',fontsize=20)
    
    dx = fig.add_subplot(224)
    seob_f = seob_valid.to_frequencyseries()
    nr_f = nr_valid.to_frequencyseries()
    dx.loglog(seob_f.sample_frequencies,np.abs(seob_f),label='SEOBNREHM')
    dx.loglog(nr_f.sample_frequencies,np.abs(nr_f),label='RIT:'+str(nrid))
    dx.axvline(float(nr.metadata['freq-start-22-Hz-1Msun'])/mtotal[ii],ls='--',color='gray',label='NR fstart 22')
    dx.legend()
    dx.set_xlabel('Frequency / Hz')
    dx.set_ylabel('$|h_+|$',fontsize=18)
    
    #create output folder
    if not os.path.exists(output_prefix):
        os.makedirs(output_prefix)
        
    fig.savefig(output_prefix+'/RIT-'+str(nrid)+'.pdf',bbox_inches='tight')