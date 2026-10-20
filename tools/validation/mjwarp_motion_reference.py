"""Independent FP64 spatial algebra and componentwise roundoff propagation."""
import numpy as np

U = np.finfo(np.float32).eps/2
G1 = U/(1-U)


def exact(x):
    x = np.asarray(x, np.float64)
    return x, np.zeros_like(x)


def add(a, b):
    x, ex = a; y, ey = b
    return x+y, ex+ey+G1*(abs(x)+abs(y)+ex+ey)


def multiply(a, b):
    x, ex = a; y, ey = b
    return x*y, abs(x)*ey+abs(y)*ex+ex*ey+G1*(abs(x)+ex)*(abs(y)+ey)


def cross(a, b):
    x, ex = a; y, ey = b
    out = np.empty(3); err = np.empty(3)
    for k, i, j in ((0,1,2), (1,2,0), (2,0,1)):
        p = multiply((x[i],ex[i]), (y[j],ey[j])); q = multiply((x[j],ex[j]), (y[i],ey[i]))
        out[k], err[k] = add(p, (-q[0],q[1]))
    return out, err


def motion_cross(a, b):
    x, ex = a; y, ey = b
    angular = cross((x[:3],ex[:3]), (y[:3],ey[:3]))
    linear = add(cross((x[3:],ex[3:]), (y[:3],ey[:3])), cross((x[:3],ex[:3]), (y[3:],ey[3:])))
    return np.r_[angular[0],linear[0]], np.r_[angular[1],linear[1]]


def force_cross(a, b):
    x, ex = a; y, ey = b
    angular = add(cross((x[:3],ex[:3]), (y[:3],ey[:3])), cross((x[3:],ex[3:]), (y[3:],ey[3:])))
    linear = cross((x[:3],ex[:3]), (y[3:],ey[3:]))
    return np.r_[angular[0],linear[0]], np.r_[angular[1],linear[1]]


def spatial(i):
    i = np.asarray(i, np.float64)
    a = np.zeros((6,6)); a[:3,:3] = [[i[0],i[3],i[4]], [i[3],i[1],i[5]], [i[4],i[5],i[2]]]
    h = np.array([[0,-i[8],i[7]], [i[8],0,-i[6]], [-i[7],i[6],0]])
    a[:3,3:] = h; a[3:,:3] = h.T; a[3:,3:] = np.eye(3)*i[9]
    return a


def matvec(a, v):
    x, ex = v; value = np.zeros(6); error = np.zeros(6)
    for j in range(6):
        value, error = add((value,error), multiply(exact(a[:,j]), (x[j],ex[j])))
    return value, error


def velocity(c):
    nv, nb = c['nv'], c['nbody']
    cv, ce = np.zeros((2,nb,6)), np.zeros((2,nb,6))
    dot, de = np.zeros((2,nv,6)), np.zeros((2,nv,6))
    for w in range(2):
        for b in range(1, nb):
            p = c['parent'][b]; v = (cv[w,p].copy(), ce[w,p].copy()); dof = int(c['dadr'][b])
            first = int(c['jadr'][b]); count = int(c['jnum'][b])
            for j in range(first, first+count):
                kind = c['joint'][j]
                if kind == 0:
                    for d in range(dof,dof+3):
                        v = add(v, multiply(exact(c['cdof'][w,d]), exact(c['qvel'][w,d])))
                    dof += 3
                width = 3 if kind in (0,1) else 1
                for d in range(dof,dof+width):
                    dot[w,d], de[w,d] = motion_cross(v, exact(c['cdof'][w,d]))
                for d in range(dof,dof+width):
                    v = add(v, multiply(exact(c['cdof'][w,d]), exact(c['qvel'][w,d])))
                dof += width
            cv[w,b], ce[w,b] = v
    return {'cvel':cv, 'cdof_dot':dot}, {'cvel':ce, 'cdof_dot':de}


def bias(c, motion=None, motion_error=None):
    nb, nv = c['nbody'], c['nv']
    cv = c['cvel'].astype(np.float64) if motion is None else motion['cvel']
    dot = c['cdof_dot'].astype(np.float64) if motion is None else motion['cdof_dot']
    cv_e = np.zeros_like(cv) if motion_error is None else motion_error['cvel']
    dot_e = np.zeros_like(dot) if motion_error is None else motion_error['cdof_dot']
    ca, cae, cf, cfe = (np.zeros((2,nb,6)) for _ in range(4))
    q, qe = np.zeros((2,nv)), np.zeros((2,nv))
    for w in range(2):
        if not c['gravity_disabled']:
            ca[w,0,3:] = -c['gravity'][w % len(c['gravity'])].astype(np.float64)
        for b in range(1,nb):
            p = c['parent'][b]; acc = (ca[w,p].copy(),cae[w,p].copy())
            for d in range(int(c['dadr'][b]), int(c['dadr'][b]+c['dnum'][b])):
                acc = add(acc, multiply((dot[w,d],dot_e[w,d]), exact(c['qvel'][w,d])))
            ca[w,b], cae[w,b] = acc
            inertia = spatial(c['cinert'][w,b]); v = (cv[w,b],cv_e[w,b])
            cf[w,b], cfe[w,b] = add(matvec(inertia,acc), force_cross(v,matvec(inertia,v)))
        # Include world publication, as pinned MJWarp does. Accumulation order
        # can vary among siblings; an absolute-contribution majorant bounds it.
        raw = cf[w].copy(); raw_e = cfe[w].copy(); major = abs(raw)+raw_e; count = np.ones(nb,int)
        for b in reversed(range(1,nb)):
            p = c['parent'][b]; cf[w,p] += cf[w,b]; major[p] += major[b]; count[p] += count[b]
            cfe[w,p] += cfe[w,b]
        for b in range(nb):
            gamma = (count[b]-1)*U/(1-(count[b]-1)*U)
            cfe[w,b] += gamma*major[b]
        for d in range(nv):
            body = int(c['dof_body'][d]); total = exact(0.)
            for k in range(6):
                total = add(total, multiply(exact(c['cdof'][w,d,k]), (cf[w,body,k],cfe[w,body,k])))
            q[w,d], qe[w,d] = total
    return {'cacc':ca, 'cfrc_int':cf, 'qfrc_bias':q}, {'cacc':cae, 'cfrc_int':cfe, 'qfrc_bias':qe}


def reference(c, phase):
    if phase == 'com_vel':
        return velocity(c)
    if phase == 'bias':
        return bias(c)
    mv, me = velocity(c); bv, be = bias(c,mv,me)
    return mv|bv, me|be


def check_arrays(outputs, values, errors):
    checks = []
    for name, out in outputs.items():
        expected = values[name]; error = abs(out.astype(np.float64)-expected)
        limit = 8*errors[name]
        fractions = np.divide(error,limit,out=np.zeros_like(error),where=limit>0)
        checks.append(dict(field=name, max_error=float(error.max()), maximum_budget_fraction=float(fractions.max()),
            passed=bool(np.isfinite(out).all() and np.all(error<=limit))))
    return checks
