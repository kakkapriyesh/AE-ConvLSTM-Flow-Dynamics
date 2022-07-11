import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
import math
from utils.scale import MinMaxScale_data
import numpy as np


'''Problem formulation is taken from the work of Fourier neural operator but this is trying to train the network with
-out data : https://arxiv.org/abs/2010.08895, I call this problem as vorticity dissipation problem'''

class LapLaceFilter2d(object):
    """
    Smoothed Laplacian 2D, assumes periodic boundary condition.
    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        super().__init__()
        self.dx = dx
        # no smoothing
        WEIGHT_3x3 = torch.FloatTensor([[[[0, 1, 0],
                                          [1, -4, 1],
                                          [0, 1, 0]]]]).to(device)
        # smoothed
        WEIGHT_3x3 = torch.FloatTensor([[[[1, 2, 1],
                                          [-2, -4, -2],
                                          [1, 2, 1]]]]).to(device) / 4.

        WEIGHT_3x3 = WEIGHT_3x3 + torch.transpose(WEIGHT_3x3, -2, -1)

       # print(WEIGHT_3x3)

        WEIGHT_5x5 = torch.FloatTensor([[[[0, 0, -1, 0, 0],
                                          [0, 0, 16, -0, 0],
                                          [-1, 16, -60, 16, -1],
                                          [0, 0, 16, 0, 0],
                                          [0, 0, -1, 0, 0]]]]).to(device) / 12.
        if kernel_size == 3:
            self.padding = _quadruple(1)
            self.weight = WEIGHT_3x3
        elif kernel_size == 5:
            self.padding = _quadruple(2)
            self.weight = WEIGHT_5x5

    def __call__(self, u):
        """
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            div_u(torch.Tensor): [B, C, H, W]
        """
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), 
            self.weight, stride=1, padding=0, bias=None) / (self.dx**2)
        return u.view(u_shape)


class SobelFilter2d(object):
    """
    Sobel filter to estimate 1st-order gradient in horizontal & vertical 
    directions. Assumes periodic boundary condition.
    Args:
        dx (float): spatial discretization, assumes dx = dy
        kernel_size (int): choices=[3, 5]
        device (PyTorch device): active device
    """
    def __init__(self, dx, kernel_size=3, device='cpu'):
        self.dx = dx
        # smoothed central finite diff
        WEIGHT_H_3x3 = torch.FloatTensor([[[[-1, 0, 1],
                                            [-2, 0, 2],
                                            [-1, 0, 1]]]]).to(device) / 8.  # as per equation

        # larger kernel size tends to smooth things out
        WEIGHT_H_5x5 = torch.FloatTensor([[[[1, -8, 0, 8, -1],
                                            [2, -16, 0, 16, -2],
                                            [3, -24, 0, 24, -3],
                                            [2, -16, 0, 16, -2],
                                            [1, -8, 0, 8, -1]]]]).to(device) / (9*12.)
        if kernel_size == 3:
            self.weight_h = WEIGHT_H_3x3
            self.weight_v = WEIGHT_H_3x3.transpose(-1, -2)
            self.weight = torch.cat((self.weight_h, self.weight_v), 0)
            self.padding = _quadruple(1)
        elif kernel_size == 5:
            self.weight_h = WEIGHT_H_5x5
            self.weight_v = WEIGHT_H_5x5.transpose(-1, -2)
            self.padding = _quadruple(2)        

    def __call__(self, u):
        """
        Compute both hor and ver grads
        Args:
            u (torch.Tensor): (B, C, H, W)
        Returns:
            grad_u: (B, C, 2, H, W), grad_u[:, :, 0] --> grad_h
                                     grad_u[:, :, 1] --> grad_v
        """
        # (B*C, 1, H, W)
        u_shape = u.shape
        u = u.view(-1, 1, *u_shape[-2:])
        u = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return u.view(*u_shape[:2], *u.shape[-3:])

    def grad_h(self, u):
        """
        Get image gradient along horizontal direction, or x axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            ux (torch.Tensor): [B, C, H, W] calculated gradient
        """
        # print(self.padding)
        # k=torch.rand(1,1,3,3)
        # print(k)
        # print(u.shape)
        # z=F.pad(u, (self.padding), mode='circular')
        # print(z.shape)
        ux = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_h, 
                        stride=1, padding=0, bias=None) / (self.dx)
        return ux
    
    def grad_v(self, u):
        """
        Get image gradient along vertical direction, or y axis.
        Perioid padding before conv.
        Args:
            u (torch.Tensor): [B, C, H, W]
        Returns:
            uy (torch.Tensor): [B, C, H, W] calculated gradient
        """
        uy = F.conv2d(F.pad(u, self.padding, mode='circular'), self.weight_v,   #self weight v is a kernal
                        stride=1, padding=0, bias=None) / (self.dx)
        return uy

class Burger2DIntegrate(object):
    '''
    Performs time-integration of the 2D Burger equation
    Args:
        dx (float): spatial discretization
        nu (float): hyper-viscosity
        grad_kernels (list): list of kernel sizes for first, second and forth order gradients
        device (PyTorch device): active device
    '''
    def __init__(self, dx, nu=1.0, grad_kernels=[3, 3], device='cpu'):
        
        self.nu = nu
        

        # Create gradients
        self.grad1 = SobelFilter2d(dx, kernel_size=grad_kernels[0], device=device)
        self.grad2 = LapLaceFilter2d(dx, kernel_size=grad_kernels[1], device=device)

    def backwardEuler(self, uPred, uPred0, dt):
        """
        Time integration of the 2D Burger system using implicit euler method
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """
        grad_ux = self.grad1.grad_h(0.5*uPred[:,:1,:,:]**2)
        grad_uy = self.grad1.grad_v(uPred[:,:1,:,:])

        grad_vx = self.grad1.grad_h(uPred[:,1:,:,:])
        grad_vy = self.grad1.grad_v(0.5*uPred[:,1:,:,:]**2)

        div_u = self.nu * self.grad2(uPred[:,:1,:,:])
        div_v = self.nu * self.grad2(uPred[:,1:,:,:])

        burger_rhs_u = -grad_ux - uPred[:,1:,:,:]*grad_uy + div_u
        burger_rhs_v = -uPred[:,:1,:,:]*grad_vx - grad_vy + div_v

        ustar_u = uPred0[:,:1,:,:] + dt * burger_rhs_u
        ustar_v = uPred0[:,1:,:,:] + dt * burger_rhs_v

        return torch.cat([ustar_u, ustar_v], dim=1)


    def crankNicolson(self, Un1, Un, dt, dx,scale_data_vort,scale_data_sf):
        """
        Time integration of the 2D Burger system using crank-nicolson
        Args:
            uPred (torch.Tensor): predicted quantity at t+dt #1,3,64,64 B,(W,u,v)
            uPred0 (torch.Tensor): prior timestep/input at t
            dt (float): delta t
        Returns:
            ustar (torch.Tensor): u integrated to time-step t+dt
        """

        wn=Un[:,0]  #B,1,64,64
        sfn=Un[:,1] #B,1,64,64
        wn1=Un1[:,0]    #B,1,64,64
        sfn1=Un1[:,1]  #B,1,64,64
        #viscn=Un[:,2].detach().cpu().numpy()
       # viscn1=Un1[:,2].detach().cpu().numpy()
        wn=wn.detach().cpu().numpy()
        sfn=sfn.detach().cpu().numpy()
        wn1=wn1.detach().cpu().numpy()
        sfn1=sfn1.detach().cpu().numpy()
        wn=scale_data_vort.inv_transform(wn)
        sfn=scale_data_sf.inv_transform(sfn)
        wn1=scale_data_vort.inv_transform(wn1)
        sfn1=scale_data_sf.inv_transform(sfn1)
        w0=np.expand_dims(wn,0)
        sf0=np.expand_dims(sfn,0)
        w=np.expand_dims(wn1,0)
        sf=np.expand_dims(sfn1,0)
        w0=torch.from_numpy(w0).cuda()
        sf0=torch.from_numpy(sf0).cuda()
        w=torch.from_numpy(w).cuda()
        sf=torch.from_numpy(sf).cuda()
    #     w=uPred[:,0,:,:]
    #     sf=uPred[:,1,:,:]
    #    # v=uPred[:,2,:,:]
        w=torch.unsqueeze(w,1)
        sf=torch.unsqueeze(sf,1)
    #     #v=torch.unsqueeze(v,1)
    #     w0=uPred0[:,0,:,:]
    #     sf0=uPred0[:,1,:,:]
      #  v0=uPred0[:,2,:,:]
        w0=torch.unsqueeze(w0,1)
        sf0=torch.unsqueeze(sf0,1)
        #v0=torch.unsqueeze(v0,1)
        # print(uPred0[:,0,:,:])
        # print(torch.max(w))
        # print(uPred0[:,1,:,:])
        # print(torch.max(u))
        # print(uPred0[:,2,:,:])
        # print(torch.max(v))
        grad_sfx = self.grad1.grad_h(sf) #gradux
        grad_sfy = self.grad1.grad_v(sf)
        grad_sfx0 = self.grad1.grad_h(sf0) #gradux
        grad_sfy0 = self.grad1.grad_v(sf0)
        grad_wx = self.grad1.grad_h(w)
        grad_wy = self.grad1.grad_v(w)  ##??

        div_w = self.nu * self.grad2(w)
        
        s=64

        grad_wx0 = self.grad1.grad_h(w0)
        grad_wy0 = self.grad1.grad_v(w0)  
        div_w0 = self.nu * self.grad2(w0)
        div_sf = self.grad2(sf)
        ##forcing term
        t = torch.linspace(0, 1, s+1, device=torch.device('cuda'))
        t = t[0:-1]

        X,Y = torch.meshgrid(t, t)
        f = 0.1*(torch.sin(2*math.pi*(X + Y)) + torch.cos(2*math.pi*(X + Y)))
        Force=torch.zeros(w.shape,device=torch.device('cuda'))
        Force[:,:]=f
       
        VD_rhs_w = div_w-grad_sfy*grad_wx+grad_sfx *grad_wy+Force
        VD_rhs_w0 = div_w0-grad_sfy0*grad_wx0+grad_sfx0*grad_wy0+Force
        
        wstar = w0 + 0.5 * dt * (VD_rhs_w + VD_rhs_w0)
        sf0=F.pad(sf0, (1,1,1,1), mode='circular')
        sf_avg=(sf0[:,:,:-2,1:-1]+sf0[:,:,2:,1:-1]+sf0[:,:,1:-1,2:]+sf0[:,:,1:-1,:-2])
        sfstar =(dx*dx*wstar[:,:,:,:]+sf_avg)*0.25

      
        wstar=(wstar).detach().cpu().numpy()
        sfstar=(sfstar).detach().cpu().numpy()
        wstar=np.expand_dims(wstar,1)
        sfstar=np.expand_dims(sfstar,1)
        # viscn=np.expand_dims(viscn,1)
        wstar=scale_data_vort.transform(wstar)
        sfstar=scale_data_sf.transform(sfstar)
        wstar=torch.from_numpy(wstar)
        sfstar=torch.from_numpy(sfstar)
        wstar=torch.unsqueeze(wstar,0)
        sfstar=torch.unsqueeze(sfstar,0)
        Ustar=torch.cat((wstar,sfstar),0).float().cuda()
        # Ustar=torch.unsqueeze(Ustar,0).float().cuda() #B,1,3,64,64
        # Ustar=torch.unsqueeze(Ustar,0).float().cuda() #B,1,3,64,64
        return (Ustar)