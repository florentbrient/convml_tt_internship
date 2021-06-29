import netCDF4 as nc
import numpy as np
import os
import scipy.stats as st
import operator
from collections import OrderedDict
import constants as CC
import scipy as sp
from scipy import ndimage as ndi
import time
from skimage import measure
from scipy.spatial.distance import pdist
from matplotlib import pyplot as plt


def stats(tab):
    return [np.mean(tab),np.min(tab),np.max(tab),np.std(tab),np.median(tab),np.percentile(tab,5),np.percentile(tab,95)]

def do_cyclic(label, choice=(-2,-1), rename=True) :
    if isinstance(choice,int): choice=(choice,)
    if not isinstance(choice,tuple): raise NameError(choice+" is not a valid choice for cyclic conditions.\nUse None or a tuple")
    objects = label
    shape   = objects.shape
    listObj = np.unique(objects[objects>0]).tolist()
    nbr0    = len(listObj)
    tmp     = np.moveaxis(objects, list(choice), range(-len(choice),0)) # swap cyclic axes to last axes
    dim     = len(shape)
    # reshape to 4D if needed
    if dim==4 :
      n0,n1,n2,n3 = shape
    elif dim==3 :
      n1,n2,n3 = shape
      n0 = 1
      tmp = tmp[None,:,:,:]
    elif dim==2 :
      n2,n3 = shape
      n0 = 1 ; n1 = 1
      tmp = tmp[None,None,:,:]
    for i0 in range(n0) :
      for i1 in range(n1) :
        for i2 in range(n2) :
          # last dim is cyclic
          if (tmp[i0,i1,i2,n3-1] and tmp[i0,i1,i2,0] and (tmp[i0,i1,i2,n3-1]!=tmp[i0,i1,i2,0])) :
              num = tmp[i0,i1,i2,n3-1]
              tmp[tmp==num] = tmp[i0,i1,i2,0]
              if rename : tmp[tmp==listObj.pop(-1)]=num
        if len(choice)==2 :
          for i3 in range(n3) :
            # also one before last dim is cyclic
            if (tmp[i0,i1,n2-1,i3] and tmp[i0,i1,0,i3] and (tmp[i0,i1,n2-1,i3]!=tmp[i0,i1,0,i3])) :
              num = tmp[i0,i1,n2-1,i3]
              tmp[tmp==num] = tmp[i0,i1,0,i3]
              if rename : tmp[tmp==listObj.pop(-1)]=num
    tmp     = tmp.reshape(shape)
    objects = np.moveaxis(tmp, range(-len(choice),0), list(choice))
    nbr     = len(np.unique(objects[objects>0]))
    #print "\t', nbr0 - nbr, 'objects on the borders'
    #self.nbr=nbr
    return objects

""" Statistical functions for binary cloud masks. """
""" Extracted from Typhon website                 """
def get_cloudproperties(cloudmask, connectivity=1):
    """Calculate basic cloud properties from binary cloudmask.

    Note:
        All parameters are calculated in pixels!!

    See also:
        :func:`skimage.measure.label`:
            Used to find different clouds. 
        :func:`skimage.measure.regionprops`:
            Used to calculate cloud properties.

    Parameters:
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).

    Returns:
        list:
            List of :class:`RegionProperties`
            (see :func:`skimage.measure.regionprops`)
    """
    cloudmask[np.isnan(cloudmask)] = 0

    labels = measure.label(cloudmask, connectivity=connectivity)
    #print labels.shape
    #plt.contourf(cloudmask);plt.colorbar();plt.show()
    #plt.contourf(labels);plt.title('cloud properties');plt.colorbar();plt.show()
    return measure.regionprops(labels)


def neighbor_distance(cloudproperties, mindist=0):
    """Calculate nearest neighbor distance for each cloud.
       periodic boundaries

    Note: 
        Distance is given in pixels.

    See also: 
        :class:`scipy.spatial.cKDTree`:
            Used to calculate nearest neighbor distances. 

    Parameters: 
        cloudproperties (list[:class:`RegionProperties`]):
            List of :class:`RegionProperties`
            (see :func:`skimage.measure.regionprops` or
            :func:`get_cloudproperties`).
        mindist
            Minimum distance to consider between centroids.
            If dist < mindist: centroids are considered the same object

    Returns: 
        ndarray: Nearest neighbor distances in pixels.
    """
    centroids = [prop.centroid for prop in cloudproperties]
    indices   = np.arange(len(centroids))
    neighbor_distance = np.zeros(len(centroids))
    centroids_array = np.asarray(centroids)

    for n, point in enumerate(centroids):
        #print n, point
        # use all center of mass coordinates, but the one from the point
        mytree = sp.spatial.cKDTree(centroids_array[indices != n])
        dist, indexes = mytree.query(point,k=len(centroids)-1)
        #print n,dist
        #print n,point,indexes,dist
        #ball  =  mytree.query_ball_point(point,mindist)
        distsave=dist[dist>mindist]   
        #print distsave

        #if abs(centroids_array[indexes[0]][0]-point[0])>100.:
        #  print centroids_array[indexes[0]]
        #  print n,point#,[centroids_array[ij] for ij in indexes]

        neighbor_distance[n] = distsave[0]

    return neighbor_distance

def iorg(neighbor_distance, cloudmask):
    """Calculate the cloud cluster index 'I_org'.

    See also: 
        :func:`scipy.integrate.trapz`:
            Used to calculate the integral along the given axis using
            the composite trapezoidal rule.

    Parameters: 
        neighbor_distance (list or ndarray): Nearest neighbor distances. 
            Output of :func:`neighbor_distance`. 
        cloudmask (ndarray): 2d binary cloud mask.

    Returns:
        float: cloud cluster index I_org.

    References: 
        Tompkins, A. M., and A. G. Semie (2017), Organization of tropical 
        convection in low vertical wind shears: Role of updraft entrainment, 
        J. Adv. Model. Earth Syst., 9, 1046-1068, doi:10.1002/2016MS000802.
    """
    nn_sorted = np.sort(neighbor_distance)
    NL        = len(neighbor_distance)
 
    nncdf = np.array(range(NL)) / float(NL)
    #print len(neighbor_distance),nncdf,nn_sorted
    
    # theoretical nearest neighbor cumulative frequency
    # distribution (nncdf) of a random point process (Poisson)
    lamb = (NL /
            float(cloudmask.shape[0] * cloudmask.shape[1]))
    nncdf_poisson = 1 - np.exp(-lamb * np.pi * nn_sorted**2)

    return sp.integrate.trapz(y=nncdf, x=nncdf_poisson)

def scai(cloudproperties, cloudmask, connectivity=1):
    """Calculate the 'Simple Convective Aggregation Index (SCAI)'.  

    The SCAI is defined as the ratio of convective disaggregation
    to a potential maximal disaggregation.

    See also: 
        :func:`scipy.spatial.distance.pdist`:
            Used to calculate pairwise distances between cloud entities. 
        :func:`scipy.stats.mstats.gmean`:
            Used to calculate the geometric mean of all clouds in pairs. 

    Parameters:
        cloudproperties (list[:class:`RegionProperties`]):
            Output of function :func:`get_cloudproperties`. 
        cloudmask (ndarray): 2d binary cloud mask.
        connectivity (int):  Maximum number of orthogonal hops to consider
            a pixel/voxel as a neighbor (see :func:`skimage.measure.label`).
        mask (ndarray): 2d mask of non valid pixels.

    Returns:
        float: SCAI.

    References: 
        Tobin, I., S. Bony, and R. Roca, 2012: Observational Evidence for 
        Relationships between the Degree of Aggregation of Deep Convection, 
        Water Vapor, Surface Fluxes, and Radiation. J. Climate, 25, 6885-6904,
        https://doi.org/10.1175/JCLI-D-11-00258.1

    """
    centroids = [prop.centroid for prop in cloudproperties]

    # number of cloud clusters
    N = len(centroids)

    # potential maximum of N depending on cloud connectivity
    if connectivity == 1:
        chessboard = np.ones(cloudmask.shape).flatten()
        # assign every second element with "0"
        chessboard[np.arange(1, len(chessboard), 2)] = 0
        # reshape to original cloudmask.shape
        chessboard = np.reshape(chessboard, cloudmask.shape)
        # inlcude NaNmask
        chessboard[np.isnan(cloudmask)] = np.nan
        N_max = np.nansum(chessboard)
    elif connectivity == 2: # it doesn't work
        #chessboard = np.ones(cloudmask.shape).flatten() # add by Florent
        chessboard[np.arange(1, cloudmask.shape[0], 2), :] = 0
        chessboard = np.reshape(chessboard, cloudmask.shape)
        chessboard[np.isnan(cloudmask)] = np.nan
        N_max = np.sum(chessboard)
    else:
        raise ValueError('Connectivity argument should be `1` or `2`.')

    # distance between points (center of mass of clouds) in pairs
    di = pdist(centroids, 'euclidean')
    # order-zero diameter
    D0 = sp.stats.mstats.gmean(di)

    # characteristic length of the domain (in pixels): diagonal of box
    L = np.sqrt(cloudmask.shape[0]**2 + cloudmask.shape[1]**2)

    return N / N_max * D0 / L * 1000

def adjust_spines(ax, spines):
    for loc, spine in ax.spines.items():
        if loc in spines:
            spine.set_position(('outward', 0))  # outward by 10 points
            spine.set_smart_bounds(True)
        else:
            spine.set_color('none')  # don't draw spine

    # turn off ticks where there is no spine
    if 'left' in spines:
        ax.yaxis.set_ticks_position('left')
    else:
        # no yaxis ticks
        ax.yaxis.set_ticks([])

    if 'bottom' in spines:
        ax.xaxis.set_ticks_position('bottom')
    else:
        # no xaxis ticks
        ax.xaxis.set_ticks([])

def timing(f):
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        #print '%s function took %0.3f ms' % (f.func_name, (time2-time1)*1000.0)
        return ret
    return wrap

def nc_dataset_list(files):
    data = {}
    for ss in files.keys():
        #print files
        #print ss
        #print files[ss]
        data[ss]=nc.Dataset(files[ss],'r')
    return data

def mkdir(path):
   try:
     os.mkdir(path)
   except:
     pass

def mask999(data):
    NPV = 999.
    data = np.ma.masked_values(data, NPV)
    return data

def mask0(data):
    NPV = 0.
    data = np.ma.masked_values(data, NPV)
    return data

def createoffset(offset):
    offset['LWP']=1000.
    rho0=1.14; RLVTT=2.5e6;RCP=1004.
    offset['E0']=rho0*RLVTT
    offset['Q0']=rho0*RCP
    offset['RC']=1000.
    offset['RV']=1000.
    offset['RT']=1000.
    offset['RCT']=1000.
    offset['RVT']=1000.
    offset['DTHRAD']=86400.
    offset['RNPM']=1000.
    return offset

def readvalues(file,typ=None,over=None):
   if typ is None:
     typ = '\r'
   if over is None:
     over= 1
   f = open(file, 'r'); data = f.read().split(typ)
   dataout = [ij.split('\t') for ij in data[over:]];f.close()
   return dataout

def resiz(tmp): # should be removed by pre-treatment
    tmp = np.squeeze(tmp)
    return tmp

def removebounds(tmp):
   if len(tmp.shape) == 1.:
     tmp = tmp[1:-1]
   elif len(tmp.shape) == 3.:
     tmp = tmp[1:-1,1:-1,1:-1]
   else:
     #print 'Problem removebounds'
     pass
   return tmp

def cond2plot(case,prefix,boxch=False):
  boxes = None
  if boxch:
    offbox = 15
    boxes  = [0,200,0,200] # x1,x2,y1,y2
    boxes  = [ij+offbox for ij in boxes]

  xycas = {}
  xycas['IHOP'] = {'008':([50,250],[410,410])} #[[x1,x2],[y1,y2]]
  try:
    xy = xy[case][prefix]
  except:
    #print 'Error in xy for ',case,prefix
    xy = None
  return boxes,xy


def selboxes(tmp,boxes,idx=None):
   if len(tmp.shape) == 1. and idx is not None:
     tmp = tmp[idx[0]:idx[1]]
   elif len(tmp.shape) == 3.:
     idxy=(boxes[2],boxes[3])
     idxx=(boxes[0],boxes[1])
     tmp = tmp[:,idxy[0]:idxy[1],idxx[0]:idxx[1]]
   else:
     #print 'Problem selboxes ',tmp.shape
     pass
   return tmp


def anomcalc(tmp,ip=0):
    mean  = np.mean(np.mean(tmp,axis=ip+2),axis=ip+1)
    if ip == 0:
      mean    = np.repeat(np.repeat(mean[ :,np.newaxis, np.newaxis],tmp.shape[ip+1],axis=ip+1),tmp.shape[ip+2],axis=ip+2)
    data = tmp-mean
    return data
 
def giveminmaxflux(vars):
   # vars need to be 2D at least
   levels,vmin,vmax,vdiff = None,None,None,None
   if vars[1] == "WT" or vars[1] == "W":
     if vars[0] == "THLM":
       #vmin,vmax,vdiff = -0.04,0.01,0.01  # old
       vmin,vmax,vdiff = -0.02,0.005,0.005
     if vars[0] == "RNPM":
       vmin,vmax,vdiff = -0.005,0.02,0.005

   if vmin is not None:
     nb     = abs(vmax-vmin)/vdiff+1
     levels = [vmin + float(ij)*vdiff for ij in range(int(nb))]
   return levels

def giveminmax(var,aa,offset):
    levels,vmin,vmax,vdiff,cmap = None,None,None,None,None
    if var == "WT" or var == "W":
      if aa=='Anom' or aa=='Mean':
        vmin,vmax,vdiff = -0.8,0.8,0.2 #0.05
    elif var == "WTMEAN":
      if aa=='Anom' or aa=='Mean':
        vmin,vmax,vdiff = -0.3,0.3,0.05
    elif var[0:2] == "TH": # or var == "THLM" or var == "THV":
      if aa=='Anom':
        #vmin,vmax,vdiff = -3.0,3.0,0.25
        vmin,vmax,vdiff = -0.5,0.5,0.1
    elif var == "LWP":
      if aa=='Mean':
        vmin,vmax,vdiff = 0.00,0.14,0.002
      if aa=='Anom':
        vmin,vmax,vdiff = -0.08,0.08,0.005
    elif var == "SVT004" or var == "SVT001":
      if aa=='Mean':
        vmin,vmax,vdiff = 0,26,2
      if aa=='Anom':
        vmin,vmax,vdiff = -12,12,4
    elif var == "SVT005" or var == "SVT002":
      if aa=='Mean':
        vmin,vmax,vdiff = 0,800,100
      if aa=='Anom':
        vmin,vmax,vdiff = -400,400,50
    elif var == "SVT006" or var == "SVT003":
      if aa=='Mean':
        vmin,vmax,vdiff = 0,900,100
      if aa=='Anom':
        vmin,vmax,vdiff = -200,200,50
    elif var == "RVT" or var == "RNPM":
      if aa=='Anom':
        vmin,vmax,vdiff = -6e-4,6e-4,1e-4
    elif var == "RCT":
      if aa=='Mean':
        vmin,vmax,vdiff = 0,5e-4,5e-5 #0,5e-4,1e-5 (used for 3D)
      if aa=='Anom':
        cmap = 'RdBu'
        vmin,vmax,vdiff = -4e-4,4e-4,1e-4
    elif var == "DTHRAD":
      if aa=='Anom':
        vmin,vmax,vdiff = -10,10,1
        vmin,vmax,vdiff = [ij/offset[var] for ij in (vmin,vmax,vdiff)]
    elif var == "DIVUV":
      if aa=='Anom' or aa=='Mean':
        vmin,vmax,vdiff = -0.02,0.02,0.005 #0.05
    elif var == "UT" or var == "VT":
      if aa=='Anom':
        vmin,vmax,vdiff = -2.,2.,0.5 #0.05


    if vmin is not None:
      if var in offset.keys():
        vmin,vmax,vdiff = [ij*offset[var] for ij in (vmin,vmax,vdiff)]
      nb     = abs(vmax-vmin)/vdiff+1
      levels = [vmin + float(ij)*vdiff for ij in range(int(nb))]
      #print var,aa,levels

      
    return levels,vmin,vmax,vdiff,cmap


def selectdata(data,zyx,x,y,z,xy=None):
    tmpdata = [x,y,z]
    nbsum   = sum([ij==None for ij in tmpdata])
    title   = ''

    if len(data.shape)==3 and (nbsum != 3 or nbsum != 0): # number of None
      if xy is not None:
        axis = [zyx[0],np.arange(xy.shape[1])]
        tmp  = np.zeros((len(axis[0]),len(axis[1])))
        for ij in range(xy.shape[1]):
          tmp[:,ij]=data[:,xy[1,ij],xy[0,ij]]
        data = tmp
      elif x is not None:
        data=data[:,:,x];axis=[zyx[0],zyx[1]];title+='_x'+str(x)
        if y is not None:
           data=data[:,y];axis=[zyx[0]];title+='_y'+str(y)
        if z is not None:
           data=data[z,:];axis=[zyx[1]];title+='_z'+str(z)
      elif y is not None:
        data=data[:,y,:];axis=[zyx[0],zyx[2]];title+='_y'+str(y)
        if z is not None:
           data=data[z,:];axis=[zyx[2]];title+='_z'+str(z)
      elif z is not None:
        data=data[z,:,:];axis=[zyx[1],zyx[2]];title+='_z'+str(z)
    elif len(data.shape)==2:
      axis=[zyx[1],zyx[2]];title+='_z'+str(z) # 2D is x,y data
    elif len(data.shape)==1:
      axis=[zyx[0]] # 1D is z data
    else:
      data = None; axis = None
    #print data.shape,axis
    return data,axis,title

def selectdiag(zyx,xy1,xy2):
    # diagonal that start on (x1,y1) and ends on (x2,y2)
    # xy1 : [x1,y1]
    # make diagonal
    #print  xy2[0]==xy1[0],xy2[0],xy1[0],xy2[1],xy1[1]
    xx   = np.mgrid[xy1[0]:xy2[0]:len(zyx[-1])*2*1j] # a lot of points
    if xy2[0]==xy1[0]:
      aa = 0
      yy = np.mgrid[xy1[1]:xy2[1]:len(zyx[-2])*2*1j] # a lot of points
    else:
      aa   = float(xy2[1]-xy1[1])/float(xy2[0]-xy1[0])
      bb   = xy1[1]-aa*xy1[0]
      yy   = aa*xx+bb
    #print aa
    xy   = np.zeros((2,len(xx)))*np.nan
    xorig= np.arange(len(zyx[2]))
    yorig= np.arange(len(zyx[1]))
    ik   = -1
    for ij in range(len(xx)):
      ik += 1
      xy[0,ik] = xorig[near(xorig,xx[ij])] 
      xy[1,ik] = yorig[near(yorig,yy[ij])]
      if ik>0:
        if (xy[0,ik]==xy[0,ik-1]) and (xy[1,ik]==xy[1,ik-1]):
          ik = ik-1
    xy = xy[:,:ik+1]
    return xy
      

# Find nearrest point 
def near(array,value):
  idx=(abs(array-value)).argmin()
  return idx+1 # +1 because it starts at index=0

# sign character
def signch(data):
  signch='p'
  if np.sign(data)<0:
    signch='m'
  return signch
      
def unitch(case):
    switcher = {
            'WT': "m/s",
            'RCT': "kg/kg",
            'RVT': "kg/kg",
            'THT': "K",
            'TKET': "m2/s2",
            'SVT004': "-",
    }
    return switcher.get(case, "Invalid units")


def findbasetop(data,z,epsilon):
    # Find cloud base and cloud top
    # Define as the first layer where RCT (ql) > epsilon
    cloudbase = np.nan; cloudtop = np.nan;
    for ij in range(len(z)):
      if np.isnan(cloudbase)  and data[ij] > epsilon:
         cloudbase = ij
      if ~np.isnan(cloudbase) and np.isnan(cloudtop)  and data[ij] < epsilon:
         cloudtop  = ij
    #print 'base,top : ',cloudbase,cloudtop,data*1000.
    return cloudbase,cloudtop

def cloudinfo(nx,ny,z,rct,epsilon):
    # Find cloud base and cloud top and middle of cloud

    # First computation : Use the mean 
    meanql  = np.mean(np.mean(rct,axis=2),axis=1)
    cloudbase,cloudtop = findbasetop(meanql,z,epsilon)
    """
    print cloudbase,cloudtop
    print 'cloudbase : ij='+str(cloudbase)+' for z='+str(z[cloudbase])
    print 'cloudtop  : ij='+str(cloudtop)+' for z='+str(z[cloudtop]) 
    """
    # Second computation : Use all data and compute its mean
    cloudbase0 = np.zeros((ny,nx))
    cloudtop0  = np.zeros((ny,nx))
    for ix in range(nx):
      for iy in range(ny):
         data = rct[:,iy,ix]
         cloudbase0[iy,ix],cloudtop0[iy,ix] = findbasetop(data,z,epsilon)     
    
    """
    print np.nanmean(cloudbase0[:])     
    print np.nanmean(cloudtop0[:])
    """
    cloudbase2 = np.nanmean(cloudbase0[:]) # real
    cloudtop2  = np.nanmean(cloudtop0[:]) # real
    # Make integer
    cloudbase2 = int(cloudbase2)
    cloudtop2  = int(cloudtop2)+1
    
    # Choice the second computation
    baseandtop = 1 # hard choice
    #baseandtop = 2 # hard choice
    if baseandtop == 2:
       cloudbase = cloudbase2
       cloudtop  = cloudtop2
    elif baseandtop == 3:
       cloudbase = min(cloudbase,cloudbase2)
       cloudtop  = min(cloudtop,cloudtop2) # not sure, max or min

    cloudmiddle  = int(round((cloudbase+cloudtop)/2))
    """
    print 'Compute final :'
    print 'cloudbase : ij='+str(cloudbase)+' for z='+str(z[cloudbase])
    print 'cloudtop  : ij='+str(cloudtop)+' for z='+str(z[cloudtop])
    """
    return cloudbase,cloudmiddle,cloudtop,cloudbase0,cloudtop0


def qs(temp,pres,order) :
  esat  = CC.es0 * np.exp(-(CC.RLVTT*CC.mv/CC.RU)*(1./temp - 1./CC.T0) )
  qstmp = CC.epsilon * esat/ (pres - esat)
  if order >= 1 :
    dqsP = -1*qstmp/pres # change of qsat for a change of P, with fixed T
    dqsT = CC.RLVTT*CC.epsilon*qstmp/ (CC.RD*temp*temp)# change of qsat for a change of T, with fixed P
  else :
    dqsP = 0; dqsT =0
  if order >=2 :
    d2qsP = 2*qstmp/(pres*pres)  # change of dqsat/dP for a change of P, with fixed T
    d2qsT = CC.RLVTT *CC.epsilon*CC.epsilon*qstmp*(2/temp  -1 )/(CC.RD*temp*temp*temp ) # change of dqsat/dT for a change of T, with fixed P
  else :
    d2qsP = 0; d2qsT =0

  return qstmp,dqsP,dqsT,d2qsP,d2qsT


def findlcl(q0,t0,temp,rvt,pres,z):
    # Find Lifting Condensation Level
    # surface humidity q0 (2D)
    # temp 3D, pres 3D
    # z 3D
    gradsec = -0.0098 # K/m z in meter
    gradhum = -0.0065 # K/m z in meter
    order = 0
    ss    = pres.shape
    q2D   = np.repeat(q0[np.newaxis,:,:],ss[0],axis=0)
    lcl   = np.ones((ss[1],ss[2]))*np.nan
    lfc   = np.ones((ss[1],ss[2]))*np.nanmax(z,axis=0) #*np.nan
    for i1 in range(ss[1]):
      for i2 in range(ss[2]):
        conditionlcl = True
        conditionlfc = True
        i0 = -1
        while(conditionlfc and i0<ss[0]-1):
           i0 +=1
           if conditionlcl:
             t1     = t0[i1,i2]+gradsec*z[i0,i1,i2]
             qsat1  = qs(t1,pres[i0,i1,i2],order)[0]
           else:
             gradhum= -CC.RG*(1.0+(CC.RLVTT*rvt[i0,i1,i2])/(CC.RD*t1)) \
                     / (CC.RCP+(pow(CC.RLVTT,2.0)*rvt[i0,i1,i2])/(CC.RV*pow(t1,2.0)))
             #t1     = tlcl+gradhum*(z[i0,i1,i2]-lcl[i1,i2])
             t1     += gradhum*(z[i0,i1,i2]-z[i0-1,i1,i2])
             if i1==45 and i2==45:
                #print z[i0,i1,i2],gradhum,tlcl,t1,temp[i0,i1,i2]
                pass

           if not conditionlcl and t1 > temp[i0,i1,i2]:
             conditionlfc=False
             lfc[i1,i2] = z[i0,i1,i2]
             #print i1,i2,i0,lcl[i1,i2],lfc[i1,i2]
           if conditionlcl and qsat1 < q0[i1,i2]:
             conditionlcl=False
             lcl[i1,i2] = z[i0,i1,i2]
             tlcl       = t1
             #print i1,i2,i0,lcl[i1,i2]
           #print i1,i2,i0,t1,temp[i0,i1,i2]
    return lcl,lfc

def makethv(theta,qv,ql):
   return theta*(1.0+0.61*qv/1000.-ql/1000.)

def makethl(theta,ql):
   return theta-(CC.RLVTT/CC.RCP)*ql/1000.

def makeths1(thlm,rnpm):
   a1    = 5.87
   QT    = rnpm/(np.ones((rnpm.shape))+rnpm)
   return thlm * np.exp(a1 * QT)

def makeths2(thlm,rv,rnpm):
    a1    = 5.87; a2 = -0.46
    rstar = np.ones((rv.shape))*0.0124
    QT    = rnpm/(np.ones((rnpm.shape))+rnpm)
    QV    = rv/(np.ones((rnpm.shape))+rnpm)
    tmp   = thlm * np.exp( a1 * QT)\
                 * np.exp( a2 * np.log(QV/rstar) * QT )\
                 * np.exp( a2 * (QT-QV) )
    return tmp
    


def group_variables(data, files, varread, varwrite, offset, resizb=0):
    # Create the variable vars with all information in files
    vars = {}
    extraname = []; minusname =[]
    for ss in files.keys():
      vars[ss] = {}
      for v in np.arange(len(varread)):
        #print ss,v,varread[v]
        try:
          tmp    = data[ss][varread[v]][:]
        except:
          tmp    = None
        #if 'SV' in varread[v]: 
        #  print 'SV1 ',tmp.shape,data[ss][varread[v]].shape
        if resizb and tmp is not None:
          tmp    = resiz(tmp)

        tmpsave    = None
        varwriteall = [varwrite[v]]
        if tmp is not None:
         if len(tmp.shape)==3: # traceurs
          varwriteall=[]
          minusname += [varwrite[v]]
          for ij in range(tmp.shape[0]):
            aa = varwrite[v].split('_')
            #print 'aa ',aa 
            varwriteall += [aa[0]+'_'+aa[1]+str(ij+1)+'_'+aa[2]]
          extraname += varwriteall
          tmpsave    = tmp

        ik = -1
        for namevar in varwriteall:
          ik+=1
          if tmpsave is not None:
            tmp = tmpsave[ik,:,:]
          offch   = [ij in namevar  for ij in offset.keys()]
          if sum(offch) and tmp is not None: # namevar in offset.keys() and tmp is not None:
            #print offch,np.where(offch)[0][0],offset.keys()[np.where(offch)[0][0]]
            tmp = tmp*offset[offset.keys()[np.where(offch)[0][0]]]
          vars[ss][namevar] = tmp
    return vars,extraname,minusname


def quadrants(data,data2=None,data3=None,threshold=0.0,threshold2=0.0,threshold3=0.0):
   # Make quadrants or octants
   # data should be one arrow (e.g. 40000 points)
   # By default
   # data  is WT
   # data2 is SVT004
   # data3 is SVT006
   # Outputs : slct(4D index), frac0,frac1 (fraction of quadrants), colors (index), colorall (each color info)
   thr   = [-1*abs(threshold),abs(threshold)]
   thr2  = [-1*abs(threshold2),abs(threshold2)]
   thr3  = [-1*abs(threshold3),abs(threshold3)]

   idx1a = (data >= thr[1]);idx1b = (data <= thr[0]) #updraft/downdraft
   idx1  = [idx1a,idx1b]
   idx2a,idx2b,idx3a,idx3b = None,None,None,None
   if data2 is not None:
     idx2a = (data2 >= thr2[1]);idx2b = (data2 <= thr2[0])
   if data3 is not None:
     idx3a = (data3 >= thr3[1]);idx3b = (data3 <= thr3[0])
   
   idx1   = [idx1a,idx1b]; idx2 = [idx2a,idx2b]; idx3 = [idx3a,idx3b];
   if data2 is not None:
     slct   = quadindex(idx1,idx2)
   else:
     slct   = idx1
   if data3 is not None:
     slct   = octaindex(idx1,idx2,idx3)

   #print 'sum ',quad.shape,sum(quad)
   #print 'Sum : ',[100*sum(ij)/len(data) for ij in quad],len(data)

   frac0,frac1     = percentquad(data,slct)
   colors,colorall = makecolors(data,slct)
   return slct,frac0,frac1,colors,colorall

def quadindex(idx1,idx2):
    # updraft that bring SVT004 anomaly
    idxA = idx1[0][:] & idx2[0][:]
    # updraft that has lower than average SVT004
    idxB = idx1[0][:] & idx2[1][:]
    # downdraft that has higher than average SVT004
    idxC = idx1[1][:] & idx2[0][:]
    # downdraft that bring clear air downwards
    idxD = idx1[1][:] & idx2[1][:]
    # Quadrant index
    quad = [idxA,idxB,idxC,idxD]
    return quad

def octaindex(idx1,idx2,idx3):

    # updraft that bring SVT004 anomaly + positive cloud-top tracer anomaly
    idxA = idx1[0][:] & idx2[0][:] & idx3[0][:]
    # updraft that has lower than average SVT004 + positive cloud-top tracer anomaly
    idxB = idx1[0][:] & idx2[1][:] & idx3[0][:]
    # downdraft that has higher than average SVT004 + positive cloud-top tracer anomaly
    idxC = idx1[1][:] & idx2[0][:] & idx3[0][:]
    # downdraft that bring clear air downwards + positive cloud-top tracer anomaly
    idxD = idx1[1][:] & idx2[1][:] & idx3[0][:]

    # updraft that bring SVT004 anomaly + negative cloud-top tracer anomaly
    idxE = idx1[0][:] & idx2[0][:] & idx3[1][:]
    # updraft that has lower than average SVT004 + negative cloud-top tracer anomaly
    idxF = idx1[0][:] & idx2[1][:] & idx3[1][:]
    # downdraft that has higher than average SVT004 + negative cloud-top tracer anomaly
    idxG = idx1[1][:] & idx2[0][:] & idx3[1][:]
    # downdraft that bring clear air downwards + negative cloud-top tracer anomaly
    idxH = idx1[1][:] & idx2[1][:] & idx3[1][:]

    # Octant index
    #print 'idxA ',idxA.shape # nx*ny
    octa = [idxA,idxB,idxC,idxD,idxE,idxF,idxG,idxH]
    return octa

def percentquad(data,quad):
   # Give relative fraction of quadrant
   # frac0 : relative to the grid-box
   # frac1 : relative the selection
   # size of data?
   ss    = len(data)
   tmp   = [sum(ij) for ij in quad]
   frac0 = [100.0*ij/ss for ij in tmp] 
   if sum(tmp)!=0:
     frac1 = [100.0*ij/sum(tmp) for ij in tmp]
   else:
     frac1 = None 

   return frac0,frac1


def makecolors(data,quad): #data4=None,threshold=0.0):
   # Make colors bar for coloring scatter plot in plotdistrib
   # By default
   # data is Wt

   colorblack         = (0.6,0.6,0.6) # grey
   color1,color2      = (0.,0.,1.),(1.,0.,0.) # blue if negative and red if positive
   colorall           = [color2,color1]
   colors             = np.array([colorblack]*len(data))
   if len(quad) == 2:
     colors[quad[0][:]] = colorall[0]
     colors[quad[1][:]] = colorall[1]
   elif len(quad) == 4:
     #print 'Lengths ',len(data)
     color3,color4      = (0.,0.8,1.),(1.,0.5,0.) # bleu ciel : (-,+) ; orange (+,-)
     colorall           = [color2,color4,color3,color1]
     # updraft that bring SVT004 anomaly
     colors[quad[0][:]] = colorall[0]
     # updraft that has lower than average SVT004
     colors[quad[1][:]] = colorall[1]
     # downdraft that has higher than average SVT004
     colors[quad[2][:]] = colorall[2]
     # downdraft that bring clear air downwards
     colors[quad[3][:]] = colorall[3]
   elif len(quad) == 8:
       # As before, but with negative cloud-top tracer anomaly
       # (w+,s1+,s2+),(w+,s1-,s2+),(w-,s1+,s2+),(w-,s1-,s2+),(w+,s1+,s2-),(w+,s1-,s2-),(w-,s1+,s2-),(w-,s1-,s2-)
       # the closer to Park or Davini
       color1,color2,color3,color4 = (1.,1.,0.),(1.,0.7,0.7),(0.2,0.4,0.1),(0.6,1.0,0.6)  # yellow, pink, dark green, light green
       color5,color6,color7,color8 = (1.,0.,0.),(1.,0.5,0.),(0.,0.8,1.),(0.,0.,1.) # red,orange,light blue,blue
       # new
       #color1,color2,color3,color4 = (1.,1.,0.),(0.9,0.9,0.9),(0.2,1.0,0.),(0.2,0.6,0)  # light yellow, white, light green, green
       #color5,color6,color7,color8 = (1.,0.,0.),(0.9,0.4,1.),(0.,0.6,1.),(0.,0.,1.) # red,light red/purple,light blue,blue
       colorall  = [color1,color2,color3,color4,color5,color6,color7,color8]
       for ij in range(len(quad)):
         colors[quad[ij][:]] = colorall[ij]
         
   return colors,colorall
 

def infos(case,sens,prefix='000',vtype='V0301',concat=''):
   #files={}
   files = OrderedDict() #dict([(sens[i],'i') for i in range(len(sens))])
   # local path with simulations
   path0 = '/cnrm/tropics/user/brientf/MESONH/'
   path0 = path0 + case +'/'
   # list with the names of files
   for ss in sens:
     #print 'ss ',ss, case
     # path of the selected experience
     path=path0+ss+'/'
     # name of the file (e.g. sel_HTFIR.1.V0301.000.nc4)
     #vtype = 'V0301'
    
     #prefix1 = '000'
     #if prefix is not 'None':
     #  prefix1 = prefix
     prefix += concat+".nc4"

     # Too heavy file, pre-selection before
     # e.g. ncks -O -v XHAT,YHAT,ZHAT,RHOREFZ,RCT,THLM HTFIR.1.V0301.012.nc4 HTFIR.1.V0301.012.nc4.out
     prefix2 = prefix
     if ss == 'L25.6km':
       prefix2 = prefix #+ '.out'

     if ss == 'SMALL':
       prefix2 = prefix #+ '.MNH' #'.save' #'.test2'

     if ss == 'SMALR':
       prefix2 = prefix + '.MOCR.FIRE' #'.T150' 

     if (ss == 'REF' or ss == 'L25.6km') and case == 'FIRE':
       name = 'HTFIR'
     else:
       name = ss

     filename = name+".1."+vtype+"."+prefix2

     files[ss]=path+filename
     #print 'files ',files

   return path0,files

def makepdf(tmprh,tmpcf,ltsrange):
    # Binning information
    ltstable = np.arange(ltsrange[0],ltsrange[1]+ltsrange[2],ltsrange[2])+ltsrange[2]/2

    kde      = st.gaussian_kde(tmprh)
    pdftmp   = kde(ltstable)
    pdftmp   = 100.*pdftmp/sum(pdftmp)

    statstmp, bin_edges, binnumber = st.binned_statistic(tmprh, tmpcf, 'mean', bins=len(ltstable),range=(ltsrange[0],ltsrange[1]))
    stdtmp   = [ np.std(tmpcf[binnumber==ij]) for ij in np.arange(1,len(ltstable)+1) ]
    #print ' binnumber ',binnumber
    #for ij in np.arange(1,len(ltstable)+1):
    #  print ij,np.std(tmpcf[binnumber==ij]),tmpcf[binnumber==ij]
    
    return pdftmp,statstmp,stdtmp

def do_unique(tmp):
    tmp[tmp>0]=1
    return tmp

def do_delete(objects,nbmin,rename=True) :
     listObj = np.unique(objects[objects>0]).tolist()
     if len(listObj):
       nbmax   = max(listObj)
       #print 'nbmax  ', nbmax
       while (len(listObj)):
         num = listObj.pop(0)                 
         nbcels = len(objects[objects==num])
         if (nbcels < nbmin):
           objects[objects==num] = 0
           if rename and len(listObj) : 
             lastnum = listObj.pop(-1)
             while (len(objects[objects==lastnum])<nbmin):
               objects[objects==lastnum]=0
               if (len(listObj)) : lastnum = listObj.pop(-1)
               else : break
             objects[objects==lastnum] = num 
             listObj=[num]+listObj
       nbr = len(np.unique(objects))-1 # except 0
       #print '\t', nbmax - nbr, 'objects were too small'
     else:
       nbr = 0
     return objects,nbr

def do_delete2(objects,mask,nbmin,rename=True):
    nbmax   = np.max(objects)
    #print nbmax,nbmin
    objects = timing(delete_smaller_than(mask,objects,nbmin))
    #print np.max(objects),len(np.unique(objects))
    if rename :
        labs = np.unique(objects)
        objects = np.searchsorted(labs, objects)
    nbr = len(np.unique(objects))-1 # except 0
    #print '\t', nbmax - nbr, 'objects were too small'
    return objects,nbr

def delete_smaller_than(mask,obj,minval):
  #print np.max(mask)
  #print np.max(obj)
  sizes = sp.ndimage.sum(mask,obj,np.unique(obj[obj!=0]))
  del_sizes = sizes < minval
  #print sizes,del_sizes
  del_cells = np.unique(obj[obj!=0])[del_sizes]
  #print del_cells
  objlog = np.ones((obj.shape),dtype=bool)
  #objlog = [np.logical_and(objlog,(obj!=ij)) for ij in del_cells] 
  #data4=[np.where(obj==ij,0,1) for ij in del_cells]
  #print objlog
  for cell in del_cells :
    #print cell
    objlog = np.logical_and(objlog,(obj!=cell))
    #obj[obj==cell] = 0
  obj = obj*objlog
  return obj

