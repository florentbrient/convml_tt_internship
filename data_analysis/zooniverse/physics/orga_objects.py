# -*- coding: utf-8 -*-

import sys
sys.path.append('/cnrm/tropics/user/brientf/MESONH/scripts/')
from netCDF4 import Dataset
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Circle
import tools as tl
from itertools import chain
from copy import deepcopy
from scipy import ndimage as ndi
import time
from skimage import measure
import netCDF4 as ncdf


def format(value):
    return "%9.3f" % value

#__all__ = [
#    'get_cloudproperties',
#    'neighbor_distance',
#    'iorg',
#    'scai',
#]

def findzplot(cas,hour,cloudlev):
  zplot = {}
  zplot['ARMCU']   = {'008':{'cloudbase':21,'cloudtop':86}}
  zplot['IHOP']    = {'008':{'cloudbase':15,'cloudtop':33}, \
                      '010':{'cloudbase':15,'cloudtop':33}}
  zplot['RICO']    = {'006':{'cloudbase':12,'cloudtop':67}}
  zplot['BOMEX01'] = {'010':{'cloudbase':28,'cloudtop':74}}
  zplot['FIRE']    = {'012':{'cloudbase':11,'cloudtop':62}, \
                      '024':{'cloudbase':11,'cloudtop':62}}
  print zplot
  print zplot[cas]
  print zplot[cas][hour]
  print zplot[cas][hour][cloudlev]
  try:
    zz = zplot[cas][hour][cloudlev]
  except:
    zz = 30
  print zz
  return zz

def runningmean(OBJ,N):
   # Use convolution for calculating a running mean in every column
   mode = 'same' # provide the same size of column
   xmin = 0.5
   ss   = OBJ.shape #lev,y,x
   tmp  = np.float32(OBJ.reshape((ss[0],ss[1]*ss[2])))
   print ss
   for ij in range(ss[1]*ss[2]):
     tmp[:,ij] = np.convolve(tmp[:,ij], np.ones((N,))/N, mode=mode)
   tmp[tmp>=xmin] =1
   tmp[tmp<xmin]  =0
   return tmp.reshape(ss)

def replacename(nameobj):
  svt0 = ['SVT001','SVT002','SVT003']
  svt1 = ['SVT004','SVT005','SVT006']
  for ss,svt in enumerate(svt0):
    nameobj=nameobj.replace(svt,svt1[ss])
  return nameobj

def write(ncdfOut,size,data,variables,data1D,var1D) :
  dset = ncdf.Dataset(ncdfOut,'w')
  # Define dimensions
  dset.createDimension('stats',size[0])
  dset.createDimension('var',size[1])
  dset.createDimension('lev',size[2])
  dset.description = "Statistical characterization of objects population. Stats are 1. mean, 2. min, 3. max, 4. std, 5. median, 6. 5th percentile, 7. 95th percentile."
  #dimChars = ('object','stats','levels',)
  dimChars = ('stats','lev',)
  #for i in range(len(self.dimChars)):
  #    ndim = list(self.dimChars)[::-1][i]
  #    dset.createDimension(self.dimChars[ndim],self.dims[ndim])
  #    dimChars = (self.dimChars[ndim],)+dimChars
  # For each charac, create a new field and write charac
  nk=0
  #for k,func in enumerate(self.dictFuncs):
  #    nk=k+1
  #    var=dset.createVariable(func.split(' ')[0], 'f8', dimChars)
  #    var[:] = np.take(self.characs, k, axis=0)
  #    if len(func.split(' '))>1 : var.units=func.split(' ')[1]
  for k,field in enumerate(variables):
      var = dset.createVariable(field, 'f8', dimChars)
      var[:] = data[:,k,:]
      #if len(field.split(' '))>1 : var.units=field.split(' ')[1]
  for k,field in enumerate(var1D):
      var    = dset.createVariable(field, 'f8', dimChars[-1])
      var[:] = data1D[k]
  dset.close()

def main(path,cas,sens,suffix,nameobj,vmin=0,vtype='V0301',nameZ='ZHAT',use3Dlayers=True,relab=True,connectivity=2,mindist=0,nblayer=0,zplot=30):
    if __name__== "__main__" :
      file0     = path+"{cas}/{simu}/{simu}.1.{vtype}.TTTT.nc4"
      file0     = file0.format(cas=cas[0],simu=sens[0],vtype=vtype)

      pathout   = './stats/'+'d_orga/'
      pathout  += 'd_'+cas[0]+'/'
      tl.mkdir(pathout)

      namedown = ''
      if nameobj.split('_')[0] == 'updr':
        print nameobj
        xplus   = 1 
        namedown='down_SVT'+'{0:03}'.format(int(nameobj.split('_')[1][-3:])+xplus)+'_'+nameobj.split('_')[-1]
        print namedown

      use3Dchar = ''
      if not use3Dlayers:
        use3Dchar = '_2D'
        if relab:
          use3Dchar += '_relab'
      Nchar = ''
      if nblayer>0:
        Nchar = '_N'+str(nblayer)

      fileout0  = pathout+'orga_{cas}_{simu}VVVV_'+nameobj+'_TTTT_'+str(vmin)+Nchar+use3Dchar+'.txt'
      fileout0  = fileout0.format(cas=cas[0],simu=sens[0])
      vtypech   = ''
      if vtype!='V0301':
        vtypech  = '_'+vtype
      fileout0  = fileout0.replace('VVVV',vtypech)

      NT        = len(suffix)
      NBSTATS   = 7
      for ik in range(NT):
        file     = file0.replace('TTTT',suffix[ik])
        fileout  = fileout0.replace('TTTT',suffix[ik])
        fileoutnc= fileout.replace('.txt','.nc')

        f = open(fileout, 'wb')
        print file,fileout

        DATA     = Dataset(file)
        ZHAT     = tl.removebounds(DATA.variables[nameZ][:])
        NLEV     = len(ZHAT)
        nbobj    = np.zeros(NLEV)

        # 3D mask
        OBJ0     = tl.removebounds(DATA.variables[nameobj][:])
        objmask  = tl.do_unique(deepcopy(OBJ0))
        if namedown:
          OBJdown= tl.removebounds(DATA.variables[namedown][:])

        if use3Dlayers:
          # Remove small 3D objects
          OBJ0,nbr = tl.do_delete2(OBJ0,objmask,vmin,rename=True)
          if namedown:
            OBJdown,nbrd = tl.do_delete2(OBJdown,tl.do_unique(deepcopy(OBJdown)),vmin,rename=True)
        
        # Initialized data
        if ik==0:
          print NLEV
          variables  = ['ND','AREA','ECC','EQDIA','PERIM']
          # SCAI, IORG, NDMEAN, NDSTD, AREA, AREASTD, ECC, ECCSTD, EQDIA, EQDIASTD, PERIM, PERIMSTD 
          NBVAR  = 2+2*len(variables) 
          size   = (NBSTATS,len(variables),NLEV)
          data   = np.zeros((NT,NBVAR,NLEV)) # time, var, level
          tab2   = np.zeros(size)

        # Run on every vertical level
        NL  = OBJ0.shape[0]
        if nblayer>0:
          time1 = time.time()
          OBJ0  = runningmean(tl.do_unique(deepcopy(OBJ0)),nblayer)
          time2 = time.time()
          print '%s function took %0.3f s for N=%5.0f' % ("running mean", (time2-time1)*1.0, nblayer)

        for ij in range(NL):
          OBJ      = OBJ0[ij,:,:]
          if namedown:
            OBJD     = OBJdown[ij,:,:]

          if use3Dlayers:
            # rename
            labs     = np.unique(OBJ)
            OBJ      = np.searchsorted(labs, OBJ)
            if namedown:
              OBJD     = np.searchsorted(np.unique(OBJD), OBJD)
          else:
            # Remove small 2D objects 
            if relab: # with new labels
              OBJ      = measure.label(OBJ, connectivity=connectivity)
              #OBJ      = tl.do_cyclic(OBJ)
            objmask = tl.do_unique(deepcopy(OBJ))
            OBJ,nbr = tl.do_delete2(OBJ,objmask,vmin,rename=True)
            if namedown:
              if relab: # with new labels
                OBJD    = measure.label(OBJD, connectivity=connectivity)
                #OBJD    = tl.do_cyclic(OBJD)
              OBJD,nbrd = tl.do_delete2(OBJD,tl.do_unique(deepcopy(OBJD)),vmin,rename=True)
            

          nbobj[ij] = len(np.unique(OBJ))-1
          if nbobj[ij]>2: # At least 3 objects
  
            #plt.contourf(objmask);plt.show()
            #plt.contourf(OBJ);plt.show()
            #print np.max(objmask),np.sum(objmask)

            #print cloudproperties
            #OBJprop           = measure.regionprops(OBJ)
            #print 'area   : ',[prop.area for prop in OBJprop]
            #print 'ecc    : ',[prop.eccentricity for prop in OBJprop]
            #print 'eq dia : ',[prop.equivalent_diameter for prop in OBJprop]
            #print 'perim  : ',[prop.perimeter for prop in OBJprop]
            #print 'orient : ',[prop.orientation for prop in OBJprop]
            #print 'extent : ',[prop.extent for prop in OBJprop]
            #print 'fill ar: ',[prop.filled_area for prop in OBJprop]
            #print 'max int: ',[prop.max_intensity for prop in OBJprop]


            #OBJpoly   = measure.approximate_polygon(OBJ,tolerance=0.02)
            #plt.plot(OBJpoly[:,1],OBJpoly[:,0])#;plt.colorbar()
            #plt.show()

            #if use3Dlayers:
              # Make 2D stats from 3D field
              #cloudprop       = measure.regionprops(OBJ)
            #else:
              # Make stats on 2D
              #OBJ             = measure.label(objmask, connectivity=2) 
              #cloudprop       = tl.get_cloudproperties(objmask,connectivity=2)
              #cloudprop       = measure.regionprops(OBJ) #tl.get_cloudproperties(OBJ,connectivity=2)

            OBJ             = np.ma.masked_array(OBJ, mask=(OBJ==0))
            cloudprop       = measure.regionprops(OBJ)
            objmask         = tl.do_unique(deepcopy(OBJ))

            if ij==zplot:
              centroids       = [prop.centroid for prop in cloudprop]
              centroids_array = np.asarray(centroids)
              nametitle       = fileout.split('/')[-1].split('.')[0].split('_')
              title           = 'Cross Section ('+' '.join(nametitle[1:])+')'
              fig             = plt.figure()
              plt.contourf(OBJ);plt.colorbar();plt.title(title)
              s  = np.pi*pow(mindist,2.)/2. #pow(np.pi*mindist,2.)/4.
              ax = plt.scatter(centroids_array[:,1],centroids_array[:,0],color='r',s=s,facecolors='none');

              if namedown:
                 cent_down       = [prop.centroid for prop in measure.regionprops(OBJD)]
                 cent_down_array = np.asarray(cent_down)
                 print cent_down_array
                 #plt.contourf(OBJD);plt.colorbar();plt.title(title)
                 plt.scatter(cent_down_array[:,1],cent_down_array[:,0],s=s,color='b',facecolors='none');
              #e = Circle( xy=centroids_array, radius=mindist )
              #ax.add_artist(e)
              #fig.set_clip_box(ax.bbox)
              fig.savefig(fileout.replace('.txt','.png'))
              fig.savefig(fileout.replace('.txt','.pdf'))
              plt.close()
              #plt.show()

            area            = np.asarray([prop.area for prop in cloudprop])
            ecc             = np.asarray([prop.eccentricity for prop in cloudprop])
            eqdia           = np.asarray([prop.equivalent_diameter for prop in cloudprop])
            perim           = np.asarray([prop.perimeter for prop in cloudprop])

            nd              = tl.neighbor_distance(cloudprop,mindist=mindist)
            data[ik,0,ij]   = tl.scai(cloudprop, objmask, connectivity=1)
            data[ik,1,ij]   = tl.iorg(nd, objmask)

            datastats       = [nd,area,ecc,eqdia,perim]
            offset          = 2
            for k,field in enumerate(datastats):
              data[ik,2*k+offset,ij]   = np.nanmean(datastats[k])
              data[ik,2*k+offset+1,ij] = np.nanstd(datastats[k])
              tab2[:,k,ij]             = tl.stats(datastats[k])

          #print type(ZHAT)
          #print type(nbobj)
          #print ZHAT[ij],nbobj[ij]
          tab  = np.concatenate(([ZHAT[ij]],[nbobj[ij]],data[ik,:,ij]))

          for v in tab:
            f.write(str(format(v))+' ')
          f.write("\n")
        f.close()
        #print(scai_obj)
        #print(iorg_obj-0.5)

        # Write Netcdf file
        var1D  = ['levels','nbobj','SCAI','IORG']
        data1D = [ZHAT,nbobj,data[ik,0,ij],data[ik,1,ij]]
        write(fileoutnc,size,tab2,variables,data1D,var1D)

      #plt.contourf(data[:,0,:]);plt.show()
      #plt.contourf(data[:,1,:]);plt.show()

path   = "/cnrm/tropics/user/brientf/MESONH/"
cas    = ['ARMCU'] #['IHOP'] #['AYOTTE'] #['ARMCU']# ['FIRE']
sens   = ['Ru0x0'] #['24Sx0'] #['Ru0x0']# ['Ls2x0']
suffix = ['008'] #,'024','008']
#suffix = ['002','004','006','008','010']
#suffix = ['003','006','009','012','015','018','021','024'] #['002','004','006','008'] #,'012','024']
#suffix = ['001','002','003','004','005','006','007']

nameobjs = ['updr_SVT001_02','down_SVT002_02','down_SVT003_02'] #'updr_SVT001_WT_02','down_SVT003_02','down_SVT001_WT_02','down_SVT002_WT_02']
#nameobjs = ['updr_SVT004_WT_02'] #'updr_SVT004_WT_02','down_SVT006_02','down_SVT004_WT_02','down_SVT005_WT_02']
# minimum number of 3D objects
vmin   = 3200 #800 #1600 #800 #1600 #3200
# Minimum distance between objects (in pixels)
mindist = 30

if cas[0] == 'FIRE':
  vmin    = int(vmin*25*25/(50*50))
  mindist = int(mindist*25/(50))
  for ij,field in enumerate(nameobjs):
    nameobjs[ij] = replacename(field)

# Use 2D layers of 3D objects
# If False, recalculate volume from 2D layers
use3Dlayers = False
relab       = True # by default
if not use3Dlayers:
  relab = True
  xdiv  = 10. #50. #20. #100.
  vmin  = int(vmin/xdiv)

# Running mean?
nblayer=10

# Altitude for plot cross section
cloudlev = 'cloudbase'
zplot    = findzplot(cas[0],suffix[0],cloudlev) #findzplot(cas[0])

for nameobj in nameobjs:
  main(path,cas,sens,suffix,nameobj,vmin=vmin,use3Dlayers=use3Dlayers,relab=relab,mindist=mindist,nblayer=nblayer,zplot=zplot)


