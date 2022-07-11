'''This loss function genrates the data for lid driven cavity and the functions could be used to formulate loss function
for the training of network for the problem of lid-driven cavity using physics constraint'''

from warnings import resetwarnings

from numpy.core.fromnumeric import squeeze
# from nsVorticity import NavierStokesVortFD
# from nsIntegrate import NavierStokesIntegrate
# import nsMesh
import numpy as np
import torch
import matplotlib.pyplot as plt
import os

'''
######################################################################################
Things are calculated based on the discretisation given in :
https://curiosityfluids.com/2016/03/14/streamfunction-vorticity-solution-lid-driven-cavity-flow/



############################################################################################
'''
def loss_function(Un,Un1,u_top,dx,dt,nu):   #U= B,1,3,64,64

    #splitting values
    # Un=torch.squeeze(Un,1)  #U= B,3,64,64
    # Un1=torch.sqeeze(Un1,1)## need to see if this can be made with 5 straigt away
    wn=Un[:,:,0]  #B,1,64,64
    sfn=Un[:,:,1] #B,1,64,64
    wn1=Un1[:,:,0]    #B,1,64,64
    sfn1=Un1[:,:,1]  #B,1,64,64
    viscn=Un[:,:,3]
    viscn1=Un1[:,:,3]

    def crank_nicholson(wn,sfn,wn1,sfn1,dx,dt):
    # First calc. spacial gradient components using finite volume
        Fn = stream_vort(nu,wn,sfn,dx)

        Fn1 = stream_vort(nu,wn1,sfn1,dx)

        wstar = wn[:,:,1:-1,1:-1]+dt*0.5*(Fn1 + Fn)
        wstar,sfn=update_boudnaries(u_top,dx,wstar,sfn) #B,1,64,64
        sfstar=stream_lap_cfd(wstar,sfn,dx) #B,1,64,64

        Ustar=torch.cat((wstar,sfstar,viscn),1) #B,3,64,64
        Ustar=torch.unsqueeze(Ustar,1) #B,1,3,64,64

        return Ustar


    # w=np.ones((5,1,N,N))*5  # 5 is batch
    # sf=np.ones((5,1,N,N))*3

    def update_boudnaries(u_top,dx,w,sf):
        '''making the surrounding of matrix zero to assume constant stram function at the boundary'''
        sf[:,:,:,N-1]=0.0  #right
    # print (sf)
        sf[:,:,:,0]=0.0 #left
    #  print (sf)   
        sf[:,:,0,:]=0.0 #top
    #  print (sf)
        sf[:,:,N-1,:]=0.0  #bottom
    #  print (sf)
        '''top'''
        w[:,:,0,:]=(2*(sf[:,:,0,:]-sf[:,:,1,:])/(dx*dx))-2*(u_top/dx)
    # print(w)
        '''bottom'''
        w[:,:,N-1,:]=(2*(sf[:,:,N-1,:]-sf[:,:,N-2,:])/(dx*dx))
    #  print(w)
        '''left'''
        w[:,:,:,0]=(2*(sf[:,:,:,0]-sf[:,:,:,1])/(dx*dx))
    #  print(w)
        '''right'''
        w[:,:,:,N-1]=(2*(sf[:,:,:,N-1]-sf[:,:,:,N-2])/(dx*dx))
    # print(w)


        return w,sf

    '''stream vorticity formulation for inner nodes'''
    def stream_vort(nu,w,sf,dx):

        def laplace_vort(w,dx):
            ddwx= (w[:,:,:-2,1:-1]-2*w[:,:,1:-1,1:-1]+w[:,:,2:,1:-1])/(dx**2)
            ddwy= (w[:,:,1:-1,:-2]-2*w[:,:,1:-1,1:-1]+w[:,:,1:-1,2:])/(dx**2)

            l_vort=ddwx+ddwy




        # print(l_vort)
            return l_vort
        
        def d_sf(sf,dx):
            sfdx=(-sf[:,:,:-2,1:-1]+sf[:,:,2:,1:-1])/(dx*2)
            sfdy=(-sf[:,:,1:-1,:-2]+sf[:,:,1:-1,2:])/(dx*2)
        #  print(sfdx,"\n")
        #   print(sfdy,"\n")

            return sfdx,sfdy

        def d_w(w,dx):
            wdx=(-w[:,:,:-2,1:-1]+w[:,:,2:,1:-1])/(dx*2)
            wdy=(-w[:,:,1:-1,:-2]+w[:,:,1:-1,2:])/(dx*2)
        # print(wdx,"\n")
            #print(wdy,"\n")

            return wdx,wdy
        
        l_vort=laplace_vort(w,dx)
        
        sfdx,sfdy=d_sf(sf,dx)
        wdx,wdy=d_w(w,dx)
        F=-sfdy*wdx+sfdx*wdy+nu*(l_vort)
        #print(F)
        return F


    def stream_lap(w,sf,dx):

        def laplace_stream(sf,dx):
            ddsf= (sf[:,:,:-2,1:-1]-2*sf[:,:,1:-1,1:-1]+sf[:,:,2:,1:-1])/(dx**2)
            ddsf= (sf[:,:,1:-1,:-2]-2*sf[:,:,1:-1,1:-1]+sf[:,:,1:-1,2:])/(dx**2)

            sf_lap=ddsf+ddsf




            #print(sf_lap)
            return sf_lap    

    def stream_lap_cfd(w,sf,dx):


            # ddsf= (sf[:,:-2,1:-1]-2*sf[:,1:-1,1:-1]+sf[:,2:,1:-1])/(dx**2)
            # ddsf= (sf[:,1:-1,:-2]-2*sf[:,1:-1,1:-1]+sf[:,1:-1,2:])/(dx**2)

            sf_update =(dx*dx*w[:,:,1:-1,1:-1]+sf[:,:,:-2,1:-1]+sf[:,:,2:,1:-1]+sf[:,:,1:-1,2:]+sf[:,:,1:-1,:-2])*0.25




            
            return sf_update 
    # sf,w=update_boudnaries(u_top,dx,w,sf)

    def Initialize_nu(Re,u_top,length_wall):
        
        
        nu=u_top*length_wall/Re  #kinematic viscosity
        return nu

    # nu=Initialize_nu(Re,u_top,length_wall)

    Ustar= crank_nicholson(wn,sfn,wn1,sfn1,dx,dt)

    return Ustar




###################################CFD##################################
    #################### variables #######################3
u_top=1.0
length_wall=1.0
N=5
Re=10
########################################################################3
dx=1/N #64 is the mesh size

w=np.ones((5,1,N,N))*5
sf=np.ones((5,1,N,N))*3

def update_boudnaries(u_top,dx,w,sf):
    '''making the surrounding of matrix zero to assume constant stram function at the boundary'''
    sf[:,:,:,N-1]=0.0  #right
# print (sf)
    sf[:,:,:,0]=0.0 #left
#  print (sf)   
    sf[:,:,0,:]=0.0 #top
#  print (sf)
    sf[:,:,N-1,:]=0.0  #bottom
#  print (sf)
    '''top'''
    w[:,:,0,:]=(2*(sf[:,:,0,:]-sf[:,:,1,:])/(dx*dx))-2*(u_top/dx)
# print(w)
    '''bottom'''
    w[:,:,N-1,:]=(2*(sf[:,:,N-1,:]-sf[:,:,N-2,:])/(dx*dx))
#  print(w)
    '''left'''
    w[:,:,:,0]=(2*(sf[:,:,:,0]-sf[:,:,:,1])/(dx*dx))
#  print(w)
    '''right'''
    w[:,:,:,N-1]=(2*(sf[:,:,:,N-1]-sf[:,:,:,N-2])/(dx*dx))
# print(w)


    return w,sf

'''stream vorticity formulation for inner nodes'''
def stream_vort(nu,w,sf,dx):

    def laplace_vort(w,dx):
        ddwx= (w[:,:,:-2,1:-1]-2*w[:,:,1:-1,1:-1]+w[:,:,2:,1:-1])/(dx**2)
        ddwy= (w[:,:,1:-1,:-2]-2*w[:,:,1:-1,1:-1]+w[:,:,1:-1,2:])/(dx**2)

        l_vort=ddwx+ddwy




    # print(l_vort)
        return l_vort
    
    def d_sf(sf,dx):
        sfdx=(-sf[:,:,:-2,1:-1]+sf[:,:,2:,1:-1])/(dx*2)
        sfdy=(-sf[:,:,1:-1,:-2]+sf[:,:,1:-1,2:])/(dx*2)
    #  print(sfdx,"\n")
    #   print(sfdy,"\n")

        return sfdx,sfdy

    def d_w(w,dx):
        wdx=(-w[:,:,:-2,1:-1]+w[:,:,2:,1:-1])/(dx*2)
        wdy=(-w[:,:,1:-1,:-2]+w[:,:,1:-1,2:])/(dx*2)
    # print(wdx,"\n")
        #print(wdy,"\n")

        return wdx,wdy
    
    l_vort=laplace_vort(w,dx)
    
    sfdx,sfdy=d_sf(sf,dx)
    wdx,wdy=d_w(w,dx)
    F=-sfdy*wdx+sfdx*wdy+nu*(l_vort)
    #print(F)
    return F


def stream_lap(w,sf,dx):

    def laplace_stream(sf,dx):
        ddsf= (sf[:,:,:-2,1:-1]-2*sf[:,:,1:-1,1:-1]+sf[:,:,2:,1:-1])/(dx**2)
        ddsf= (sf[:,:,1:-1,:-2]-2*sf[:,:,1:-1,1:-1]+sf[:,:,1:-1,2:])/(dx**2)

        sf_lap=ddsf+ddsf




        #print(sf_lap)
        return sf_lap    

def stream_lap_cfd(w,sf,dx):


        # ddsf= (sf[:,:-2,1:-1]-2*sf[:,1:-1,1:-1]+sf[:,2:,1:-1])/(dx**2)
        # ddsf= (sf[:,1:-1,:-2]-2*sf[:,1:-1,1:-1]+sf[:,1:-1,2:])/(dx**2)

        sf_update =(dx*dx*w[:,:,1:-1,1:-1]+sf[:,:,:-2,1:-1]+sf[:,:,2:,1:-1]+sf[:,:,1:-1,2:]+sf[:,:,1:-1,:-2])*0.25




        
        return sf_update 
# sf,w=update_boudnaries(u_top,dx,w,sf)

def Initialize_nu(Re,u_top,length_wall):
    
    
    nu=u_top*length_wall/Re  #kinematic viscosity
    return nu

nu=Initialize_nu(Re,u_top,length_wall)
    
    
  


# F=stream_vort(nu,w,sf,dx)
# SL=stream_lap(w,sf,dx)

##########CFD##################
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from matplotlib import cm


ntrain=40
Re_train=[520,625,725,905,1115,1250,1500,1750,1805,1950]
#Re_train=np.linspace(500,2000,ntrain).astype(int)
print(Re_train)
u_top=1.0
length_wall=1.0
N=64
dt=0.001
iter=0
t=0
dx=1/(N -1) #64 is the mesh size
nvalid=10
vort=np.zeros((len(Re_train),100,5,1,64,64))
stream_func=np.zeros((len(Re_train),100,5,1,64,64))  #5 is batch and not sequence number .
Re=np.zeros((len(Re_train),100,5,1,64,64))

for i in range(len(Re_train)):
  t=0
  iter=0
  nu=Initialize_nu(Re_train[i],u_top,length_wall)


  w=np.ones((5,1,N,N))*0
  sf=np.ones((5,1,N,N))*0
  Re_visc=np.ones((5,1,N,N))*nu

  #w,sf=update_boudnaries(u_top,dx,w,sf)

  while(t<19.95):
      
      w_old=w
      sf_old=sf
      
      
    # print(w)
    # print(sf)
      F=stream_vort(nu,w,sf,dx)
      w[:,:,1:-1,1:-1]=w[:,:,1:-1,1:-1]+dt*F
    # print(w)

      w,sf=update_boudnaries(u_top,dx,w,sf)
      #print(w,"\n")
    # print(sf)

      # F=stream_vort(nu,w,sf,dx)
      # w[:,1:-1,1:-1]=w[:,1:-1,1:-1]+dt*F
    # print(w)
      sf[:,:,1:-1,1:-1]=stream_lap_cfd(w,sf,dx)
      
      if iter%200==0:
        T=iter/200
        Re_temp=Re_train[i]
        #stream_func[i]=Re_valid[i]
        stream_func[i,int(T),:,:,:]=sf[:,:,:]
        vort[i,int(T),:,:,:]=w[:,:,:]
        Re[i,int(T),:,:,:]=Re_visc[:,:,:]
        print(np.max(vort[i,int(T)]),"\n","t",t,"\n")
        print(np.min(w))
        # c_max = np.max(np.array(sf))
        # c_min = np.min(np.array(sf))
        # ticks = np.linspace(0, 1, 100)
        # tickLabels = np.linspace(c_min, c_max, 100)

        # fig1 = plt.figure('imshow')
        #     #img=plt.imshow(squeeze(sf), cmap="plasma")
        # plt.contourf(squeeze(sf), ticks=tickLabels, spacing='proportional', extend='both')
        # plt.colorbar(label="Like/Dislike Ratio", orientation="vertical")
        # plt.title('sf')
        # plt.show()
        print(int(T))
        print("iter",int(T),"i",i)
        if int(T)%99==0:
          c_max = np.max(np.array(sf))
          c_min = np.min(np.array(sf))
          ticks = np.linspace(0, 1, 10)
          tickLabels = np.linspace(c_min, c_max, 10)

          fig1 = plt.figure('imshow')
              #img=plt.imshow(squeeze(sf), cmap="plasma")
          plt.contourf((sf[0,0]), levels=10,ticks=tickLabels, spacing='proportional', extend='both')
          plt.colorbar(label="Like/Dislike Ratio", orientation="vertical")
          plt.title('Stream_Vorticity')
          load_dir="./Video_combined"
          file_dir = load_dir
# If director does not exist create it
          if not os.path.exists(file_dir):
                os.makedirs(file_dir)
          
          file_name=file_dir+"/RE_{:d}_{:2f}_{:2f}".format(Re_train[i],T,i)
          plt.savefig(file_name + ".png", bbox_inches='tight')
          # plt.show()
          plt.close("all")
      iter=iter+1
      #print("iter",iter,"t",t)
      t=t+dt

print("saving figures")

# file_name = "Visc_40set_RE{:d}_{:d}".format(500,2000)

# file = open(file_name, "wb")
# np.save(file_name, Re)
# file.close


# file_name = "stream_function_40set_RE{:d}_{:d}".format(500,2000)

# file = open(file_name, "wb")
# np.save(file_name, stream_func)
# file.close


# file_name = "vorticity_40set_RE{:d}_{:d}".format(500,2000)

# file = open(file_name, "wb")
# np.save(file_name, vort)
# file.close


# def velocity_to_stream_vort(dx,u):
#     udx = (u[:,:,2:,1:-1] - u[:,:,:-2,1:-1])/(2*dx[0,0,1,1]) 
#     print(udx)
#     udy = (u[:,:,1:-1,2:] - u[:,:,1:-1,:-2])/(2*dx[0,1,1,1])
#     print(udy)
#     wz = udx[:,1] - udy[:,0]
#     print(wz)

#     return wz


# w=velocity_to_stream_vort(dx,U)





