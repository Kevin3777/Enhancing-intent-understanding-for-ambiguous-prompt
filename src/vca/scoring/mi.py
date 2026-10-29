"""Theory validation only. CLIP cosine is not a true MI estimator."""
import torch

def gaussian_mi(x,y,xy):
    joint=torch.cat([torch.cat([x,xy],1),torch.cat([xy.T,y],1)],0)
    for cov in (x,y,joint):torch.linalg.cholesky(cov)  # require positive definite
    return (torch.linalg.slogdet(x)[1]+torch.linalg.slogdet(y)[1]-torch.linalg.slogdet(joint)[1])/2

def rank_one_proxy(rho,c=1.):
    rho=torch.as_tensor(rho)
    if c<=0 or (c*rho.square()>=1).any():raise ValueError('Eq12 domain: C>0 and C*rho^2<1')
    return -.5*torch.log1p(-c*rho.square())

def update(theta,mi_gradient,eta,mode='paper_literal'):
    if mode not in ('paper_literal','mi_ascent'):raise ValueError(mode)
    return theta+(-1 if mode=='paper_literal' else 1)*eta*mi_gradient
