import numpy as np
import torch


div = lambda a,b: np.divide(a, b, out=np.zeros_like(a), where=b!=0)

def depthmap_phi_lamb(R):
    def point_forward(x,y,phi1,lamb0,fov):
        rho=np.sqrt(x**2+y**2)
        c=np.arctan2(rho,1)
        sinc=np.sin(c)
        cosc=np.cos(c)

        phi=np.arcsin(cosc*np.sin(phi1)+(div(y*sinc*np.cos(phi1),rho)))
        lamb=lamb0+np.arctan2(x*sinc,rho*np.cos(phi1)*cosc-y*np.sin(phi1)*sinc)

        phi=np.where(phi<-np.pi/2,np.pi/2-phi,phi)
        lamb=np.where(lamb<-np.pi,2*np.pi+lamb,lamb)

        phi=np.where(phi>np.pi/2,-np.pi/2+phi,phi)
        lamb=np.where(lamb>np.pi,-2*np.pi+lamb,lamb)

        return phi,lamb

    H,W = R.shape[-2:]
    
    u,v = np.meshgrid(np.linspace(1,-1,W),np.linspace(1,-1,H))

    a_phi,a_theta = point_forward(u,v,0,0,(1,1))
    return a_phi, a_theta 

def depthmap_to_3D(yy,adjust_depth=None):

    a_phi, a_theta = depthmap_phi_lamb(yy)

    if adjust_depth:
        R =  yy / torch.tensor((np.cos(a_phi)*np.cos(a_theta))[None,None,:,:]).type_as(yy)
        R = torch.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)
        
    else:
        R = yy
        yy = R * torch.tensor((np.cos(a_phi)*np.cos(a_theta))[None,None,:,:]).type_as(yy)
    
    
    xx = R*torch.tensor((np.cos(a_phi)*np.sin(a_theta))[None,None,:,:]).type_as(yy)
    
    zz = R*torch.tensor(np.sin(a_phi)[None,None,:,:]).type_as(yy)
    
    return torch.cat([xx,yy,zz],axis=1)#.permute(0,2,3,1).reshape(B,H*W,3))
