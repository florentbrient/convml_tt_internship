import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
"""

Code from https://github.com/martinjanssens/cloudmetrics/tree/master/Metrics

"""
connectivity = 
areaMin = 
bc = "periodic"
bins = 
csdFit = "power"
def opensky(field, bc=None, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    os : float
        Open sky parameter, assuming a rectangular reference area.
    '''
    aOSMax = 0; aOSAv = 0    
    for i in range(field.shape[0]): # rows
        cl_ew = np.where(field[i,:] == 1)[0] # cloudy pixels
        for j in range(field.shape[1]): # cols
            if field[i,j] != 1:
                
                #FIXME for speed -> do this once and store
                cl_ns = np.where(field[:,j] == 1)[0] 
                
                ws = np.where(cl_ew < j)[0] # west side cloudy pixels
                es = np.where(cl_ew > j)[0] # east side
                ns = np.where(cl_ns < i)[0] # north side
                ss = np.where(cl_ns > i)[0] # south side
                
                # West side
                if ws.size == 0: # if no cloudy points left of this pixel
                    if bc == 'periodic' and es.size != 0:
                        w = cl_ew[es[-1]] - field.shape[1]
                    else:
                        w = 0
                else:
                    w = cl_ew[ws[-1]]
                
                # East side
                if es.size == 0:
                    if bc == 'periodic' and ws.size != 0:
                        e = cl_ew[ws[0]] + field.shape[1] - 1
                    else:
                        e = field.shape[1]
                else:
                    e = cl_ew[es[0]] - 1
                
                # North side
                if ns.size == 0: 
                    if bc == 'periodic' and ss.size != 0:
                        n = cl_ns[ss[-1]] - field.shape[0]
                    else:
                        n = 0
                else:
                    n = cl_ns[ns[-1]]
                
                # South side
                if ss.size == 0: 
                    if bc == 'periodic' and ns.size != 0:
                        s = cl_ns[ns[0]] + field.shape[0] - 1
                    else:
                        s = field.shape[0]
                else:
                    s = cl_ns[ss[0]] - 1
                
                aOS = (e - w)*(s - n) # Assuming rectangular reference form
                
                aOSAv += aOS
                if aOS > aOSMax:
                    aOSMax = aOS
                    osc    = [i,j]
                    nmax,smax,emax,wmax = n,s,e,w
    aOSAv = aOSAv / field[field == 0].size / field.size
    osMax = aOSMax / field.size

    if plot:
        plt.figure(); ax = plt.gca()
        ax.imshow(field,'gray')
        rect = patches.Rectangle((wmax,nmax),emax-wmax,smax-nmax,
                                    facecolor='none',edgecolor='C0',
                                    linewidth='3')
        ax.add_patch(rect)
        ax.scatter(osc[1],osc[0],s=100)
        ax.set_axis_off()
        ax.set_title('e: ' + str(emax) + ', w: ' + str(wmax) + ', n: ' +
                        str(nmax) + ', s: ' + str(smax))
        plt.show()
            
    return osMax, aOSAv



def cf(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    cf : float
        Clod fraction.
    '''
    cf = len(np.where(field == 1)[0]) / field.size
    
    if plot:
        plt.imshow(field,'gray')
        plt.title('Cloud fraction: '+str(round(cf,3))) 
        plt.show()
    
    return cf


def cop(field):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    COP : float
        Convective Organisation Potential.
    '''
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    area = []; xC = []; yC = []
    for i in range(num):
        props  = regions[i]
        if props.area > areaMin:
            y0, x0 = props.centroid
            xC.append(x0); yC.append(y0)
            area.append(props.area)
    area = np.asarray(area)
    pos  = np.vstack((np.asarray(xC),np.asarray(yC))).T
    nCl  = len(area)
    
    # print('Number of regions: ',pos.shape[0],'/',num)

    if len(area) < 1:
        return float("nan")

    ## COMPUTE COP (Array-based)
    # pairwise distances (handling periodic BCs)
    if bc == 'periodic':
        dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
        for d in range(field.ndim):
            # Number of pixels in original field's dimension, assuming the 
            # field was doubled
            box = field.shape[d] // 2
            pos_1d = pos[:, d][:, np.newaxis]  # shape (N, 1)
            dist_1d = sd.pdist(pos_1d)  # shape (N * (N - 1) // 2, )
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d ** 2  # d^2 = dx^2 + dy^2 + dz^2
        dist = np.sqrt(dist_sq)
    else:
        dist = sd.pdist(pos)
    dij = sd.squareform(dist)                   # Pairwise distance matrix
    dij = dij[np.triu_indices_from(dij, k=1)]   # Upper triangular (no diag)
    aSqrt = np.sqrt(area)                       # Square root of area
    Aij = aSqrt[:, None] + aSqrt[None,:]        # Pairwise area sum matrix 
    Aij = Aij[np.triu_indices_from(Aij, k=1)]   # Upper triangular (no diag)
    Vij = Aij / (dij*np.sqrt(np.pi))            # Pairwise interaction pot.
    cop = np.sum(Vij)/(0.5*nCl*(nCl-1))         # COP
    
    return cop


def csd(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    sizeExp if csdFit is power: float
        Exponent of the power law fit of the cloud size distribution
    popt if csdFit is perc : list
        List of fit parameters of the percolation fit by Ding et al (2014).
    '''
    
    # Segment
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    # Extract length scales
    area = []
    for i in range(num):
        props  = regions[i]
        if props.area > areaMin:
            area.append(props.area)
    area = np.asarray(area);
    l    = np.sqrt(area)
    
    plt.hist(area)

    # Construct histogram
    hist = np.histogram(l,bins)
    ns   = hist[0]
    
    # Filter zero bins and the first point
    ind   = np.where(ns != 0)
    nssl  = ns[ind]
    lavsl = self.lav[ind]
    nssl  = nssl[1:]
    lavsl = lavsl[1:]
    
    # Regular fit
    if csdFit == 'power':
        csd_sl, csd_int = np.polyfit(np.log(lavsl), np.log(nssl), 1)
        rSq = rSquared(np.log(lavsl),np.log(nssl), [csd_sl, csd_int])
        
        if plot:
            fig,axs = plt.subplots(ncols=2,figsize=(8.5,4))
            axs[0].imshow(field,'gray')
            axs[0].set_xticks([]); axs[0].set_yticks([])
            axs[1].scatter(np.log(lavsl), np.log(nssl),s=10,c='k')
            axs[1].plot(np.log(self.lav), 
                        csd_int+csd_sl*np.log(self.lav),c='gray')
            # axs[1].plot(np.log(lav), fPerc(lav,popt[0],popt[1],popt[2]))
            axs[1].set_xlim((np.log(self.lav[1])-0.2,
                                np.log(np.max(self.lav))+0.2))
            axs[1].set_ylim((-0.5,np.log(np.max(ns))+0.5))
            axs[1].set_xlabel(r'log $s$ [m]');
            axs[1].set_ylabel(r'log $n_s$ [-]')
            axs[1].annotate('exp = '+str(round(csd_sl,3)),(0.6,0.9),
                            xycoords='axes fraction')
            axs[1].annotate(r'$R^2$ = '+str(round(rSq,3)),(0.6,0.8),
                            xycoords='axes fraction')
            plt.show()
        
        return csd_sl
    
    # Subcritical percolation fit
    elif csdFit == 'perc':
        popt,pcov = curve_fit(fPerc,lavsl,np.log(nssl))
        if popt[0] > 0:
            popt[0] = 0
        
        if plot:
            fig,axs = plt.subplots(ncols=2,figsize=(8.5,4))
            axs[0].imshow(field,'gray')
            axs[0].set_xticks([]); axs[0].set_yticks([])
            axs[1].scatter(np.log(lavsl), np.log(nssl),s=10,c='k')
            axs[1].plot(np.log(self.lav), 
                        fPerc(self.lav,popt[0],popt[1],popt[2]))
            axs[1].set_xlim((np.log(self.lav[1])-0.2,
                                np.log(np.max(self.lav))+0.2))
            axs[1].set_ylim((-0.5,np.log(np.max(ns))+0.5))
            axs[1].set_xlabel(r'log $s$ [m]');
            axs[1].set_ylabel(r'log $n_s$ [-]')
            # axs[1].annotate('exp = '+str(round(csd_sl,3)),(0.6,0.9),
                            # xycoords='axes fraction')
            # axs[1].annotate(r'$R^2$ = '+str(round(rSq,3)),(0.6,0.8),
                            # xycoords='axes fraction')
            plt.show()
        
        return popt


def cth(field,mask, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud top height field.
    mask : numpy array of shape (npx,npx) 
        Cloud mask field.
    Returns
    -------
    ave : float
        Mean cloud top height.
    var : float
        Standard deviation of cloud top height
    ske : float
        Skewness of cloud top height distribution
    kur : float
        Kurtosis of cloud top height
        
    '''
    
    field *= mask
    
    # Filter high clouds explicitly for this computation
    field[field>self.thr] = 0 
    cthnz = field[field != 0]
    
    ave = np.mean(cthnz)
    var = np.std(cthnz)
    ske = skew(cthnz,axis=None)
    kur = kurtosis(cthnz,axis=None)
                    
    # Plotting routine
    if plot:
        bns = np.arange(1,self.thr,300)
        fig,axs = plt.subplots(ncols=3,figsize=(15,4))
        axs[0].imshow(mask,'gray'); axs[0].set_title('Cloud mask')
        a2=axs[1].imshow(field,'gist_ncar')
        axs[1].set_title('Cloud top height')
        cb = plt.colorbar(a2); 
        cb.ax.set_ylabel('Cloud top height [m]', rotation=270, labelpad=1)
        hst,_,_ = axs[2].hist(cthnz.flatten(),bns); 
        axs[2].set_title('CTH histogram')
        plt.tight_layout()
        plt.show()
    
    return ave, var, ske, kur

def cwp(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud water path field.
    Returns
    -------
    cwp : float
        Scene-integrated cloud water.
    cwpVar : float
        Variance in cloud water over entire scene.
    cwpVar : float
        Variance in cloud water in cloudy regions.
    cwpSke : float
        Skewness of water distribution in cloudy regions.
    cwpKur : float
        Kurtosis of water distribution in cloudy regions.            
    '''
    
    # Variance over entire scene
    cwpSum = np.sum(field)
    cwpVar = np.std(field,axis=None)
    cwpSke = skew(field,axis=None)
    cwpKur = kurtosis(field,axis=None)

    # Variance in cloudy regions only
    cwpMask  = field.copy(); cwpMask[field==0] = float('nan')
    varCl    = np.nanstd(cwpMask,axis=None)

    # plot
    if plot:
        cwppl = field.copy()
        ind   = np.where(field>self.pltThr); cwppl[ind] = self.pltThr
        cwppl[cwppl == 0.] = float('nan')
        fig,axs = plt.subplots(ncols=2,figsize=(8,4))
        axs[0].imshow(cwppl,'gist_ncar')
        axs[0].set_title('CWP'); axs[0].axis('off')
        axs[1].hist(field.flatten(),np.linspace(1,self.pltThr,100))
        axs[1].set_title('Histogram of in-cloud CWP')
        axs[1].set_xlabel('CWP'); axs[1].set_ylabel('Frequency')
        plt.show()
    
    return cwpSum, cwpVar, cwpSke, cwpKur, varCl

def fourrier(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    beta : float
        Directly computed spectral slope.
    betaa : float
        Bin-averaged spectral slope.
    azVar : float
        Measure of variance in the azimuthal power spectrum (anisotropy)
    lSpec : float
        Spectral length scale (de Roode et al, 2004)
    lSpecMom : float
        Spectral length scale based on moments
        (e.g. Jonker et al., 1998, Jonker et al., 2006)
    '''
    # Spectral analysis - general observations
    # Windowing   : Capturing more information is beneficial 
    #               (Planck>Welch>Hann window)
    # Detrending  : Mostly imposes unrealistic gradients
    # Using image : Less effective at reproducing trends in 2D org plane 
    # Binning     : Emphasises lower k and ignores higher k
    # Assumptions : 2D Fourier spectrum is isotropic
    #               Impact of small-scale errors is small
    
    [X,Y] = np.meshgrid(np.arange(field.shape[0]),
                        np.arange(field.shape[1]), indexing='ij')
    
    # Detrend
    if self.detrend:
        field,bDt  = detrend(field,[X,Y])
    
    # Windowing
    if bc != 'periodic':
        if self.window == 'Planck':
            field = planckRad(field)        # Planck-taper window
        elif self.window == 'Welch':
            field = welchRad(field)         # Welch window
        elif self.window == 'Hann':
            field = hannRad(field)          # Hann window  
    
    # FFT
    F       = fftpack.fft2(field)       # 2D FFT (no prefactor)
    F       = fftpack.fftshift(F)       # Shift so k0 is centred
    psd2    = np.abs(F)**2/\
                np.prod(field.shape)      # Get the energy-preserving 2D PSD
    psd1    = getPsd1D(psd2)            # Azimuthal integral-> 1D PSD
    psd1Az  = getPsd1DAz(psd2)          # Radial integral -> Sector 1D PSD
    azVar   = 2*(np.max(psd1Az[1]) -
                    np.min(psd1Az[1]))     # Spectrum anisotropy (0-1)
    
    # Direct beta
    shp  = np.min(field.shape)
    k1d     = -(fftpack.fftfreq(shp,self.dx))[shp//2:]
    k1d     = np.flip(k1d)
    beta,b0 = np.polyfit(np.log(k1d),
                            np.log(psd1),1) # Spectral slope beta
    rSqb    = rSquared(np.log(k1d),
                        np.log(psd1),
                        [beta,b0])        # rSquared of the fit
    
    # Average over bins
    k0   = np.log10(1/(shp*self.dx))
    kMax = np.log10(self.dx/2)        # Max wavenumber
    bins = np.logspace(k0,kMax,self.nBin+1)
    binsA  = np.exp((np.log(bins[1:]) + np.log(bins[:-1]))/2)
    mns = np.zeros(len(bins)-1); sts = np.zeros(len(bins-1))
    for i in range(len(bins)-1):
        imax   = np.where(k1d <  bins[i+1])[0][-1]
        imin   = np.where(k1d >= bins[i])[0]
        if len(imin) == 0:
            continue # You have gone beyond the available wavenumbers
        else:
            imin = imin[0]
        if imin == imax:
            psdi = psd1[imin]
        else:
            psdi   = psd1[imin:imax]
        mns[i] = np.mean(psdi)
        sts[i] = np.std (psdi)
    binsA = binsA[mns!=0]
    mns   = mns[mns!=0]
    
    # betaa
    if mns.shape[0] != 0:
        betaa,b0a = np.polyfit(np.log(binsA[1:-1]),np.log(mns[1:-1]),1)        # Spectral slope beta
        rSqba = rSquared(np.log(binsA[1:-1]),np.log(mns[1:-1]),[betaa,b0a])    # rSquared of the fit
    else:
        betaa,b0a = float('nan'), float('nan')

    # Spectral length scale as de Roode et al. (2004), using true median
    # sumps = np.cumsum(psd1); sumps/=sumps[-1]
    # kcrit = np.where(sumps>1/2)[0][0]
    # lSpec = 1./kcrit
    
    # Spectral length scale as de Roode et al. (2004) using ogive
    varTot = np.trapz(psd1,k1d); i = 0; vari = varTot+1
    while vari > 2./3*varTot:
        vari = np.trapz(psd1[i:],k1d[i:])
        i += 1
    kcrit = k1d[i-1]
    lSpec = 1./kcrit
    
    # Spectral length scale as Jonker et al. (2006), using moments:
    kMom     = np.trapz(psd1*k1d**self.expMom,k1d) / varTot
    lSpecMom = 1./kMom
    
    # Plotting
    if plot:
        fig,axs = plt.subplots(ncols=2,figsize=(8,4))
        axs[0].imshow(field,'gray'); axs[0].axis('off')
        axs[0].set_title('Clouds')
        # axs[1].imshow(np.log(psd2)); axs[1].axis('off')
        # axs[1].set_title('2D PSD - Anisotropy: %.3f' %azVar)
        axs[1].scatter(np.log(k1d),np.log(psd1),s=2.5,c='k')
        axs[1].plot(np.log(k1d),b0+beta*np.log(k1d),c='k')
        axs[1].scatter(np.log(binsA),np.log(mns),s=2.5,c='C1')
        axs[1].axvline(np.log(kcrit),c='grey')

        locs = axs[1].get_xticks().tolist()
        labs = [x for x in axs[1].get_xticks()]
        Dticks=dict(zip(locs,labs))
        Dticks[np.log(kcrit)] = r'$1/\Lambda$'
        locas=list(Dticks.keys()); labes=list(Dticks.values())
        axs[1].set_xticks(locas); axs[1].set_xticklabels(labes)
        
        axs[1].plot(np.log(binsA),b0a+betaa*np.log(binsA),c='C1')
        axs[1].annotate('Direct',(0.7,0.9), xycoords='axes fraction',
                        fontsize=10)
        axs[1].annotate(r'$R^2$='+str(round(rSqb,3)),(0.7,0.8), 
                        xycoords='axes fraction',fontsize=10)
        axs[1].annotate(r'$\beta=$'+str(round(beta,3)),(0.7,0.7), 
                        xycoords='axes fraction',fontsize=10)
        axs[1].annotate(r'$\Lambda=$'+str(round(lSpec,3)),(0.7,0.6), 
                        xycoords='axes fraction',fontsize=10)
        axs[1].annotate(r'$\Lambda_M=$'+str(round(lSpecMom,3)),(0.7,0.5), 
                        xycoords='axes fraction',fontsize=10)
        axs[1].annotate('Bin-averaged',(0.4,0.9), xycoords='axes fraction',
                        color='C1',fontsize=10)
        axs[1].annotate(r'$R^2$='+str(round(rSqba,3)),(0.4,0.8), 
                        xycoords='axes fraction',color='C1',fontsize=10)
        axs[1].annotate(r'$\beta_a=$'+str(round(betaa,3)),(0.4,0.7), 
                        xycoords='axes fraction',color='C1',fontsize=10)
        axs[1].set_xlabel(r'$\ln k$',fontsize=10)
        axs[1].set_ylabel(r'$\ln E(k)$',fontsize=10)
        axs[1].grid()
        axs[1].set_title('1D Spectrum')
        plt.tight_layout()
        plt.show()
    
    return beta, betaa, azVar, lSpec, lSpecMom

def fracdim(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    fracDim : float
        Minkowski-Bouligand dimension.
    '''
    Z      = (field < self.thr)               # Binary image
    p      = min(Z.shape)
    n      = 2**np.floor(np.log(p)/np.log(2)) 
    n      = int(np.log(n)/np.log(2))         # Number of extractable boxes
    sizes  = 2**np.arange(n, 1, -1)           # Box sizes
    counts = np.zeros(len(sizes))
    for s in range(len(sizes)):
        counts[s] = boxcount(Z, sizes[s])     # Non-empty/non-full box no.
        
    # Fit the relation: counts = coeffs[1]*sizes**coeffs[0]; coeffs[0]=-Nd
    coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)  
    rSq    = rSquared(np.log(sizes),np.log(counts),coeffs)
    fracDim = -coeffs[0]                   
    
    if plot:
        fig,ax=plt.subplots(ncols=2,figsize=(8.25,4))
        ax[0].imshow(field,'gray')
        ax[0].set_xticks([]); ax[0].set_yticks([])
        ax[1].loglog(sizes,counts)
        ax[1].set_title('fracDim = %.4f'%fracDim)
        ax[1].annotate('rSq: %.3f'%rSq,(0.7,0.9),xycoords='axes fraction')
        ax[1].set_xlabel('Length')
        ax[1].set_ylabel('Number of edge boxes')
        plt.show()
    
    return fracDim

def iorg(self,field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    iOrg : float
        Organisation index from comparison to inhibition nearest neighbour
        distribution.
    '''
    #fix seed for reproducible results
    random.seed(self.seed)

    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    cr = []; 
    xC = []; yC = []
    for i in range(len(regions)):
        props  = regions[i]
        if props.area > areaMin:
            y0, x0 = props.centroid
            xC.append(x0); yC.append(y0)
            cr.append(props.equivalent_diameter/2)
    
    posScene = np.vstack((np.asarray(xC),np.asarray(yC))).T
    cr       = np.asarray(cr)
    cr       = np.flip(np.sort(cr))                  # Largest to smallest
    
    # print('Number of regions: ',posScene.shape[0],'/',num)

    if posScene.shape[0] < 1:
        return float('nan')
    
    if bc == 'periodic':
        sh = [shd//2 for shd in field.shape]
        sz = np.min(sh) # FIXME won't work for non-square domains
        
        # Move centroids outside the original domain into original domain
        posScene[posScene[:,0]>=sh[1],0] -= sh[1]
        posScene[posScene[:,0]<0,0]      += sh[1]
        posScene[posScene[:,1]>=sh[0],1] -= sh[0]
        posScene[posScene[:,1]<0,1]      += sh[0]
    else:
        sh = [shd for shd in field.shape]
        sz = None
    
    nndScene  = cKDTreeMethod(posScene,sz)
    
    iOrgs = np.zeros(self.numCalcs)
    for c in range(self.numCalcs):
        # Attempt to randomly place all circles in scene without ovelapping
        i=0; placedCircles = []; placeCount = 0
        while i < len(cr) and placeCount < self.maxTries:
            new = circle(cr[i],sh)
            placeable = True
            
            # If the circles overlap -> Place again
            if checkOverlap(new,placedCircles):
                placeable = False; placeCount += 1
            
            if placeable:
                placedCircles.append(new)
                i+=1; placeCount = 0
        
        if placeCount == self.maxTries:
            # TODO should ideally start over again automatically
            print('Unable to place circles in this image') 
        else:
            if plot:
                fig1 = plt.figure(figsize=(5,5)); ax = plt.gca()
                ax.set_xlim((0,field.shape[1]));ax.set_ylim((0,field.shape[0]))
                for i in range(len(placedCircles)):
                    circ = plt.Circle((placedCircles[i].xm,placedCircles[i].yp)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].x ,placedCircles[i].yp)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].xp,placedCircles[i].yp)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].xm,placedCircles[i].y )
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].x ,placedCircles[i].y )
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].xp,placedCircles[i].y )
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].xm,placedCircles[i].ym)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].x ,placedCircles[i].ym)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                    circ = plt.Circle((placedCircles[i].xp,placedCircles[i].ym)
                                        ,placedCircles[i].r); ax.add_artist(circ)
                ax.grid(which='both')
                plt.show()
    
            ## Compute the nearest neighbour distances ##
            
            # Gather positions in array
            posRand = np.zeros((len(placedCircles),2))
            for i in range(len(placedCircles)):
                posRand[i,0] = placedCircles[i].x
                posRand[i,1] = placedCircles[i].y
            
            # If field has open bcs, do not compute nn distances using 
            # periodic bcs
            nndRand   = cKDTreeMethod(posRand,sz)
            # nndScene  = cKDTreeMethod(posScene,sz)
            
            # Old bin generation:
            # nbins = len(nndRand)+1
            # bmin = np.min([np.min(nndRand),np.min(nndScene)])
            # bmax = np.max([np.max(nndRand),np.max(nndScene)])
            # bins = np.linspace(bmin,bmax,nbins)
            
            # New:
            nbins = 10000 # <-- Better off fixing nbins at a very large number
            bins = np.linspace(0, np.sqrt(sh[0]**2+sh[1]**2), nbins)
            
            nndcdfRan = np.cumsum(np.histogram(nndRand, bins)[0])/len(nndRand)
            nndcdfSce = np.cumsum(np.histogram(nndScene,bins)[0])/len(nndScene)
                    
            ## Compute Iorg ##
            iOrg = np.trapz(nndcdfSce,nndcdfRan)  
            iOrgs[c] = iOrg
            
            if plot:
                fig,axs=plt.subplots(ncols=4,figsize=(20,5))
                
                axs[0].imshow(field,'gray')
                axs[0].set_title('Cloud mask of scene')
                
                axs[1].scatter(posScene[:,0],field.shape[0] - posScene[:,1],
                                color='k',s=5)
                axs[1].set_title('Scene centroids')
                axs[1].set_xlim((0,field.shape[1]))
                axs[1].set_ylim((0,field.shape[0]))
                asp = np.diff(axs[1].get_xlim())[0] / \
                        np.diff(axs[1].get_ylim())[0]
                axs[1].set_aspect(asp)
                
                axs[2].scatter(posRand[:,0],posRand[:,1],color='k',s=5)
                axs[2].set_title('Random field centroids')
                asp = np.diff(axs[2].get_xlim())[0] / \
                        np.diff(axs[2].get_ylim())[0]
                axs[2].set_aspect(asp)
                
                axs[3].plot(nndcdfRan,nndcdfSce,'-',color='k')
                axs[3].plot(nndcdfRan,nndcdfRan,'--',color='k')
                axs[3].set_title('Nearest neighbour distribution')
                axs[3].set_xlabel('Random field nearest neighbour CDF')
                axs[3].set_ylabel('Scene nearest neighbour CDF')
                axs[3].annotate(r'$I_{org} = $'+str(round(iOrg,3)),(0.7,0.1),
                                xycoords='axes fraction')
                asp = np.diff(axs[3].get_xlim())[0] / \
                        np.diff(axs[3].get_ylim())[0]
                axs[3].set_aspect(asp)
                plt.show()
    # print(iOrgs)
    return np.mean(iOrgs)

def iorg_poisson(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    iOrg : float
        Organisation Index.
    '''
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    xC = []; yC = []
    for i in range(len(regions)):
        props  = regions[i]
        if props.area > areaMin:
            y0, x0 = props.centroid; 
            xC.append(x0); yC.append(y0)

    pos = np.vstack((np.asarray(xC),np.asarray(yC))).T
    
    # print('Number of regions: ',pos.shape[0],'/',num)

    if pos.shape[0] < 1:
        print('No sufficiently large cloud objects, returning nan')
        return float('nan')

    ## Compute the nearest neighbour distances ##
    if bc == 'periodic':
        sh = [shd//2 for shd in field.shape]
        sz = np.min(sh) # FIXME won't work for non-square domains
        
        # Move centroids outside the original domain into original domain
        pos[pos[:,0]>=sh[1],0] -= sh[1]
        pos[pos[:,0]<0,0]      += sh[1]
        pos[pos[:,1]>=sh[0],1] -= sh[0]
        pos[pos[:,1]<0,1]      += sh[0]
        
    else:
        sh = [shd for shd in field.shape]
        sz = None
    nnScene  = cKDTreeMethod(pos,size=sz)
    # nbins = len(nnScene)+1; dx=0.01 
    nbins = 100000 # <-- Better off fixing nbins at a very large number
    bins = np.linspace(0, np.sqrt(sh[0]**2+sh[1]**2), nbins)
    nndpdfScene = np.histogram(nnScene, bins)[0]
    nndcdfScene = np.cumsum(nndpdfScene) / len(nnScene)
    
    # Poisson
    lam   = nnScene.shape[0] / (sh[0]*sh[1])
    binav = (bins[1:] + bins[:-1])/2
    nndcdfRand  = 1 - np.exp(-lam*np.pi*binav**2) 
            
    ## Compute Iorg ##
    iOrg = np.trapz(nndcdfScene,nndcdfRand)
    
    if plot:
        fig,axs=plt.subplots(ncols=3,figsize=(15,5))
        axs[0].imshow(field,'gray')
        axs[0].set_title('Cloud mask of scene')
        
        axs[1].scatter(pos[:,0],field.shape[0] - pos[:,1],
                        color='k', s=5)
        axs[1].set_title('Scene centroids')
        asp = np.diff(axs[1].get_xlim())[0] / np.diff(axs[1].get_ylim())[0]
        axs[1].set_aspect(asp)
        
        axs[2].plot(nndcdfRand,nndcdfScene,'-',color='k')
        axs[2].plot(nndcdfRand,nndcdfRand,'--',color='k')
        axs[2].set_title('Nearest neighbour distribution')
        axs[2].set_xlabel('Poisson nearest neighbour CDF')
        axs[2].set_ylabel('Scene nearest neighbour CDF')
        axs[2].annotate(r'$I_{org} = $'+str(round(iOrg,3)),(0.7,0.1),
                        xycoords='axes fraction')
        asp = np.diff(axs[2].get_xlim())[0] / np.diff(axs[2].get_ylim())[0]
        axs[2].set_aspect(asp)
        plt.show()
    
    return iOrg

def network(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    netVarDeg : float
        Variance of the network cells' degree (number of sides) 
        distribution.
    netAWPar : float
        Aboav-Wearie arrangement parameter (slope of fit between degree of
        a cell and average degree of neighbour cells).
    netCoPar : float
        Combination of netVarDeg and netAWPar
    netLPar : float
        Lewis law fit (slope of fit between degree and cloud size)
    netLCorr : float
        Correlation of the Lewis law fit
    netDefSl : float
        Slope of the 'defect' model (slope of fit between degree 
        deviation from hexagonal and normalised cloud size)
    netDegMax : float
        Intercept of the 'defect' model
    '''
    G = create_cell_graph(field,areaMin,method='scipy')

    if plot:
        fig, ax = plt.subplots(1, 1)
        ax.imshow(field); plt.axis('off')
        plot_graph(G, ax=ax)
        
        a = int(0.05 * G.graph['ngrid'])
        b = int(0.95 * G.graph['ngrid'])
        c = int(0.05 * G.graph['ngrid'])
        d = int(0.95 * G.graph['ngrid'])
        ax.add_patch(
        patches.Rectangle((a, c),
                            abs(b - a),
                            abs(d - c),
                            fill=False,
                            color='c',
                            lw=3))
    
    netVarDeg = degree_variance(G,plot)
    netAWPar  = aboav_weaire_parameter(G,plot)
    netCoPar  = coordination_parameter(netAWPar, netVarDeg)
    netLPar   = lewis_parameter(G,plot)
    netLCorr  = lewis_correlation(G,plot)
    [netDefSl,netDegMax]  = defect_parameters(G,plot)
    
    return netVarDeg, netAWPar, netCoPar, netLPar, netLCorr, netDefSl, \
            netDegMax

def objects(self,field,im, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    im    : numpy array of shape (npx,npx)
        Reference image
    Returns
    -------
    lMax : float
        Length scale of largest object in the scene.
    lMean : float
        Mean length scale of scene objects.
    eccA : float
        Area-weighted, mean eccentricity of objects, approximated as 
        ellipses
    periS : float
        Mean perimeter of an object in the scene
    '''
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    area = []; ecc  = []; peri = []
    for i in range(num):
        props  = regions[i]
        if props.area > areaMin:
            area.append(props.area)
            ecc .append(props.eccentricity)
            peri.append(props.perimeter)      
    area = np.asarray(area); ecc = np.asarray(ecc); peri = np.asarray(peri)
    # area = np.sqrt(area) <- Janssens et al. (2021) worked in l-space.
    #                         However, working directly with areas before
    #                         taking mean is more representative of pattern        
            
    # print('Number of regions: ',len(area),' / ',num)
    
    # Plotting
    if plot:
        bins = np.arange(-0.5,len(area)+1.5,1)
        fig,axs=plt.subplots(ncols=5,figsize=(15,3))
        axs[0].imshow(field,'gray'); axs[0].set_title('Cloud mask')
        axs[1].imshow(im,'gray'); axs[1].set_title('Reference image')
        axs[2].hist(area,bins); axs[1].set_title('Area')
        axs[3].hist(ecc, bins); axs[2].set_title('Eccentricity')
        axs[4].hist(peri,bins); axs[3].set_title('Perimeter')
        plt.show()

    if len(area) < 1:
        return float('nan'),float('nan'),float('nan'),float('nan'),float('nan')

    lMax    = np.sqrt(np.max(area))
    lMean   = np.sqrt(np.mean(area))
    nClouds = len(area)
    eccA    = np.sum(area*ecc)/np.sum(area)
    periS   = np.mean(peri)
    
    return lMax, lMean, nClouds, eccA, periS

def orientation(self,field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    scai : float
        Simple Convective Aggregation Index.
    '''
    
    if bc == 'periodic':
        print('Periodic BCs not implemented for orientation metric, returning nan')
        return float('nan')
    
    cov = moments_cov(field)
    if np.isnan(cov).any() or np.isinf(cov).any():
        return float('nan')

    evals,evecs = np.linalg.eig(cov)
    orie = np.sqrt(1 - np.min(evals)/np.max(evals))
    
    if plot:
        sort_indices = np.argsort(evals)[::-1]
        x_v1, y_v1 = evecs[:, sort_indices[0]]  # evec with largest eval
        x_v2, y_v2 = evecs[:, sort_indices[1]]
        evalsn = evals[sort_indices] / evals[sort_indices][0]
        
        scale = 10
        ox    = int(field.shape[1]/2)
        oy    = int(field.shape[0]/2)
        lw    = 5
        
        fig = plt.figure(); ax = plt.gca()
        ax.imshow(field,'gray')
        # plt.scatter(ox+x_v1*-scale*2,oy+y_v1*-scale*2,s=100)
        ax.plot([ox-x_v1*scale*evalsn[0], ox+x_v1*scale*evalsn[0]],
                    [oy-y_v1*scale*evalsn[0], oy+y_v1*scale*evalsn[0]],
                    linewidth=lw)
        ax.plot([ox-x_v2*scale*evalsn[1], ox+x_v2*scale*evalsn[1]],
                    [oy-y_v2*scale*evalsn[1], oy+y_v2*scale*evalsn[1]],
                    linewidth=lw)
        ax.set_title('Alignment measure = '+str(round(orie,3)))
        plt.show()
    
    return orie

def rdf(field, S, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    S     : Tuple of the input field's size (is different from field.shape
            if periodic BCs are used)
    Returns
    -------
    rdfM : float
        Maximum of the radial distribution function.
    rdfI : float
        Integral of the radial distribution function.
    rdfD : float
        Max-min difference of the radial distribution function.
    '''
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    xC = []; yC = []
    for i in range(num):
        props  = regions[i]
        if props.area > areaMin:
            yC.append(props.centroid[0])
            xC.append(props.centroid[1])
    
    pos = np.vstack((np.asarray(xC),np.asarray(yC))).T
    
    # print('Number of regions: ',pos.shape[0],'/',num)

    if pos.shape[0] < 1:
        print('No sufficiently large cloud objects, returning nan')
        return float('nan'),float('nan'),float('nan')
    
    
    # TODO set dr based on field size and object number, results are 
    # sensitive to this
    rdf, rad, tmp = pair_correlation_2d(pos, S,                                            
                                        self.rMax, self.dr, bc,
                                        normalize=True)
    rad *= self.dx
    rdfM = np.max(rdf)
    rdfI = np.trapz(rdf,rad)
    rdfD = np.max(rdf) - rdf[-1]
    
    if plot:
        axF = 'axes fraction'
        fig,axs = plt.subplots(ncols=2,figsize=(8.5,4))
        axs[0].imshow(field,'gray')
        axs[0].axis('off')
        
        axs[1].plot(rad,rdf)
        axs[1].set_xlabel('Distance')
        axs[1].set_ylabel('RDF')
        axs[1].annotate('rdfMax = %.3f' %rdfM,(0.6,0.15), xycoords=axF)
        axs[1].annotate('rdfInt = %.3f' %rdfI,(0.6,0.10), xycoords=axF)
        axs[1].annotate('rdfDif = %.3f' %rdfD,(0.6,0.05), xycoords=axF)
        plt.show()

    return rdfM, rdfI, rdfD

def scai(self,field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    D0 : float
        Mean geometric nearest neighbour distance between objects.
    scai : float
        Simple Convective Aggregation Index.
    '''
    cmlab,num  = label(field,return_num=True,connectivity=connectivity)
    regions    = regionprops(cmlab)
    
    xC = []; yC = []
    for i in range(num):
        props  = regions[i]
        if props.area > areaMin:
            y0, x0 = props.centroid
            xC.append(x0); yC.append(y0)
    pos = np.vstack((np.asarray(xC),np.asarray(yC))).T
    nCl = pos.shape[0]
    
    # print('Number of regions: ',pos.shape[0],'/',num)

    if pos.shape[0] < 1:
        print('No sufficiently large cloud objects, returning nan')
        return float('nan'), float('nan')
    
    if bc == 'periodic':
        dist_sq = np.zeros(nCl * (nCl - 1) // 2)  # to match the result of pdist
        for d in range(field.ndim):
            box = field.shape[d] // 2
            pos_1d = pos[:, d][:, np.newaxis]
            dist_1d = sd.pdist(pos_1d)
            dist_1d[dist_1d > box * 0.5] -= box
            dist_sq += dist_1d ** 2
        dist = np.sqrt(dist_sq)
    else:
        dist = sd.pdist(pos)
        
    D0   = gmean(dist)
    Nmax = field.shape[0]*field.shape[1]/2
    scai = num / Nmax * D0 / self.L * 1000
    
    # Force SCAI to zero if there is only 1 region (completely aggregated)
    # This is not strictly consistent with the metric (as D0 is 
    # objectively undefined), but is consistent with its spirit
    if pos.shape[0] == 1:
        scai = 0
    
    if plot:
        plt.imshow(field,'gray')
        plt.title('scai: '+str(round(scai,3))) 
        plt.show()
    
    return D0, scai

def tmpvar(field, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud mask field.
    Returns
    -------
    twpVar : float
        Variance ratio of cloud water in blocks of (L0,L0) to scene in 
        total.
    '''
    
    lwpBlock = blockShaped(field,self.L0,self.L0)          # Blocks (L0,L0)
    lwpAnom  = np.mean(lwpBlock,axis=(1,2))-np.mean(field) # Lwp anomaly 
                                                            # per block
    lwpStd   = np.std(lwpAnom)                             # Compute std
    twpVar   = lwpStd / np.std(field)                      # / Domain std
    
    if plot:
        lwpPlot = field.copy()
        lwpPlot[lwpPlot>self.thr] = self.thr
        lwpPlot[lwpPlot<1] = float('nan')
        aDim = int(field.shape[0]/self.L0)
        fig,axs = plt.subplots(ncols=2,figsize=(8,4))
        axs[0].imshow(lwpPlot,cmap='gist_ncar')
        axs[0].set_title('Cloud water path field')
        axs[1].imshow(lwpAnom.reshape(aDim,aDim))
        axs[1].set_title('CWP anomaly; twpVar = '+str(round(twpVar,3)))
    
        return twpVar

def woi(field,cm,verify=False, plot=False):
    '''
    Compute metric(s) for a single field
    Parameters
    ----------
    field : numpy array of shape (npx,npx) - npx is number of pixels
        Cloud water path field.
    cm : numpy array of shape (npx,npx) 
        Cloud mask field.
    Returns
    -------
    woi1 : float
        First wavelet organisation index (scale distribution).
    woi2 : float
        Second wavelet organisation index (total amount of stuff).
    woi3 : float
        Third wavelet organisation index (directional alignment).
    '''
    
    # STATIONARY/UNDECIMATED Direct Wavelet Transform
    field = pywt.pad(field, self.pad, 'periodic')
    scaleMax = int(np.log(field.shape[0])/np.log(2))
    coeffs = pywt.swt2(field,'haar',scaleMax,norm=True,trim_approx=True)
    # Bug in pywt -> trim_approx=False does opposite of its intention
    # Structure of coeffs:
    # - coeffs    -> list with nScales indices. Each scale is a 2-power of 
    #                the image resolution. For 512x512 images we have
    #                512 = 2^9 -> 10 scales
    # - coeffs[i] -> Contains three directions:
    #                   [0] - Horizontal
    #                   [1] - Vertical
    #                   [2] - Diagonal
    
    specs = np.zeros((len(coeffs),3))  # Shape (nScales,3)
    k = np.arange(0,len(specs))
    for i in range(len(coeffs)):
        if i == 0:
            ec = coeffs[i]**2
            specs[i,0] = np.mean(ec)
        else:
            for j in range(len(coeffs[i])):
                ec = coeffs[i][j]**2     # Energy -> squared wavelet coeffs
                specs[i,j] = np.mean(ec) # Domain-averaging at each scale     
    
    # Decompose into ''large scale'' energy and ''small scale'' energy
    # Large scales are defined as 0 < k < 5
    specs = specs[1:]
    specL = specs[:5,:]
    specS = specs[5:,:]
    
    Ebar  = np.sum(np.mean(specs,axis=1))
    Elbar = np.sum(np.mean(specL,axis=1))
    Esbar = np.sum(np.mean(specS,axis=1))
    
    Eld    = np.sum(specL,axis=0)
    Esd    = np.sum(specS,axis=0)
    
    # Compute wavelet organisation index
    woi1 = Elbar / Ebar
    woi2 = (Elbar + Esbar) / np.sum(cm)
    woi3 = 1./3*np.sqrt(np.sum(((Esd - Esbar)/Esbar)**2 + ((Eld - Elbar)/Elbar)**2))
    
    woi  = np.log(woi1) + np.log(woi2) + np.log(woi3)
    
    if plot:
        labs = ['Horizontal','Vertical','Diagonal']
        fig,axs = plt.subplots(ncols=2,figsize=(8,4))
        axs[0].imshow(field,'gist_ncar'); 
        axs[0].set_xticks([]); axs[0].set_yticks([])
        axs[0].set_title('CWP')
        for i in range(3):
            axs[1].plot(k[1:],specs[:,i],label=labs[i])
        axs[1].set_xscale('log')
        axs[1].set_xlabel(r'Scale number $k$')
        axs[1].set_ylabel('Energy')
        axs[1].set_title('Wavelet energy spectrum')
        axs[1].legend()
        plt.tight_layout()
        plt.show()

    if verify:
        return specs
    else:
        return woi1, woi2, woi3, woi