
import numpy as np
from pylab import *

def aop(s,axis=0):
    s1 = np.take(s,1,axis)
    s2 = np.take(s,2,axis)
    return 0.5*np.arctan2(s2,s1)

class Fresnel:
    def _vdot(a,b,axis=-1,keepdims=True):
        """sum-product along axis"""
        return np.sum(a*b,axis=axis,keepdims=keepdims)

    def _refract(ki,n,mi,mt):
        #use inner() instead of dot() for broadcasting
        kll = ki-n*Fresnel._vdot(ki,n)
        return kll - n*np.sqrt(Fresnel._vdot(ki,ki)*(mt/mi)**2 - Fresnel._vdot(kll,kll))
        #st = (mi/mt)*np.cross(np.cross(n,ki),n)
        #return st - n*np.sqrt(1-np.sum(st**2,axis=-1))

    def _transmission(ki,kt,n,mi,mt):
        tt = 2*Fresnel._vdot(ki,n,keepdims=False)
        ts = tt / Fresnel._vdot(ki + kt, n,keepdims=False)
        tp = mi*mt*tt / Fresnel._vdot(mt**2*ki + mi**2*kt, n,keepdims=False)
        return ts, tp

    def transmission(ki,n,mi,mt):
        """Propagation vector and transmission coefficients (kt, ts, tp)"""
        ki,n,mi,mt = map(np.asarray,(ki,n,mi,mt))
        kt = Fresnel._refract(ki,n,mi,mt)
        return (kt,)+Fresnel._transmission(ki,kt,n,mi,mt)

class Mueller:
    def rotator(angle):
        """Rotates Stokes Vector by angle radians from S1 to S2. This is the opposite of the Polarized Light book convention"""
        c = np.cos(2*angle)
        s = np.sin(2*angle)
        return np.array([[1,0,0,0],[0,c,-s,0],[0,s,c,0],[0,0,0,1]])

    def rotate(mat,angle):
        """Rotate a Mueller matrix from S1 to S2 by angle radians"""
        return np.dot(Mueller.rotator(angle), np.dot(mat, Mueller.rotator(-angle)))

    def polarizer(px,py,angle=0):
        a = px**2+py**2
        b = px**2-py**2
        c = 2*px*py
        m = 0.5*np.array([[a,b,0,0],[b,a,0,0],[0,0,c,0],[0,0,0,c]])
        if angle != 0:
            return Mueller.rotate(m,angle)
        else:
            return m

    def rayleigh_norm(th):
        c = np.cos(th)
        c2 = c**2
        a = (c2-1)/(c2+1)
        b = 2*c/(c2+1)
        return np.array([[1,a,0,0],[a,1,0,0],[0,0,b,0],[0,0,0,b]])

class Jones:
    def toMueller(mat):
        mat = np.array(mat,copy=False)
        A = np.array([[1,0,0,1],[1,0,0,-1],[0,1,1,0],[0,1j,-1j,0]])
        return 0.5*np.real(A.dot(np.kron(mat, mat.conj())).dot(A.T.conj()))

class Scattering:
    def vspf_fournier(th,n,mu):
        n = np.real(n)
        d = 4*np.sin(th/2)**2 / (3*(n-1)**2)
        d_180 = 4*np.sin(np.pi/2) / (3*(n-1)**2)
        v = (3-mu)/2
        dv = d**v
        d_180v = d_180**v
        d1 = 1-d
        dv1 = 1-dv

        a = 1/(4*np.pi*dv*d1**2)
        b = v*d1-dv1+(d*dv1-v*d1)*np.sin(th/2)**(-2)
        c = (1-d_180v)*(3*np.cos(th)**2 - 1)/(16*np.pi*(d_180-1)*d_180v)
        return a*b+c

def Mxform(x1,y1,x2,y2):
    """Transform a stokes vector from coordinates x1,y1 to x2,y2"""
    return Jones.toMueller([[dot(x2,x1), dot(x2, y1)], [dot(y2,x1), dot(y2,y1)]])

def norm_cross(a,b):
    c = cross(a,b)
    return c/norm(c)

def Mrotv(k,x1,x2):
    """Rotation of a wave w/ propagation vector k rotating from x-axis x1 to x2"""
    y1 = norm_cross(k,x1)
    y2 = norm_cross(k,x2)
    return Mxform(x1,y1,x2,y2)

def vector_angle(a,b):
    return arccos(sum(a*b,-1)/sqrt(sum(a**2,-1)*sum(b**2,-1)))


def oceanaop(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483):
    """Compute aop"""
    # Array of angles in radians, in the range [-pi, pi]
    return aop(oceanstokes(sun_az,sun_zen,cam_head,cam_elev,m2,npart,mu),-1)


def oceanstokes(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483):
    """Compute stokes vectors"""
    b = broadcast(sun_az,sun_zen,cam_head,cam_elev,m2,npart,mu)
    st = empty(b.shape+(4,))
    st_flat = st.reshape((-1,4))
    for i,x in enumerate(b):
        st_flat[i] = oceansim(*x)[0]
    return st

def oceansim(sun_az,sun_zen,cam_head,cam_elev=0,m2=1.33,npart=1.08,mu=3.483, debug=True):
    n = array([0,0,1]) # water surface normal vector
    m1 = 1.0 # air index of refraction
    #vector from sun:
    ki = -array([sin(sun_az)*sin(sun_zen), cos(sun_az)*sin(sun_zen), cos(sun_zen)])
    xi = norm_cross(n,ki)
    #transmitted sunlight
    #tx, ty are the transmission amplitude coefficients in the xt, yt directions
    kt,tx,ty = Fresnel.transmission(ki,n,m1,m2)
    xt = xi
    #vector to camera
    kc = -array([sin(cam_head)*cos(cam_elev),cos(cam_head)*cos(cam_elev),sin(cam_elev)])*norm(kt)
    xc = norm_cross(n, kc) #right
    yc = norm_cross(kc, xc) #up
    #vectors for scattering
    ys = norm_cross(kt, kc) # y-axis of scattering event
    xst = norm_cross(ys, kt) # x-axis of scattering event relative to transmitted sunlight
    xsc = norm_cross(ys, kc) # x-axis of scattering event relative to camera
    #Mueller matrices
    #  transmission through water surface:
    mm1 = Mueller.polarizer(tx,ty)
    #  rotate to scattering plane
    mm2 = Mrotv(kt,xt,xst)
    #  scatter
    th_s = vector_angle(kt,kc)
    #mm3 = Mocean(rad2deg(th_s)) #using Empirical ocean scattering
    mm3 = Mueller.rayleigh_norm(th_s) #normalized Rayleigh scattering matrix
    b = Scattering.vspf_fournier(th_s,npart,mu)
    #  transform to camera's horizontal and up vectors
    mm4 = Mxform(xsc,ys, xc,yc)
    #Combined: mm4 . (b*mm3) . mm2 . mm1
    m = mm4.dot(b*mm3.dot(mm2.dot(mm1)))
    #stokes vector
    s = m.dot([1,0,0,0])
    if debug:
        return s,m,(ki,xi),(kt,xt,xst),(kc,xc,xsc),(mm1,mm2,mm3,b,mm4)
    else:
        return s,m
