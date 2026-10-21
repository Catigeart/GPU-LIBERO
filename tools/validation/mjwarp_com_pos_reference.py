"""Independent FP64 CoM/spatial algebra with input-derived FP32 error bounds."""
import numpy as np
from .mjwarp_motion_reference import U, G1, exact, add, multiply, cross, spatial


def gamma(n):
    return n*U/(1-n*U)


def reference(c, phase, *, frame_errors=None):
    nb, nj, nv = c['nbody'], c['njnt'], c['nv']
    frame_errors = frame_errors or {}
    fe = lambda name: frame_errors.get(name,np.zeros_like(c[name],dtype=float))
    center = c['subtree_com'].astype(float).copy(); ce = np.zeros_like(center)
    values = {}; errors = {}
    if phase in ('subtree','composed','mass-chain'):
        for w in range(2):
            mass = c['body_mass'][w % len(c['body_mass'])].astype(float)
            weighted = c['xipos'][w].astype(float)*mass[:,None]
            we = fe('xipos')[w]*abs(mass[:,None])
            we += G1*(abs(weighted)+we)
            major = abs(weighted)+we; count = np.ones(nb,int)
            for b in range(nb-1,0,-1):
                p = c['parent'][b]; weighted[p] += weighted[b]; we[p] += we[b]; major[p] += major[b]; count[p] += count[b]
            # Any sibling order and all ancestor additions are bounded by the
            # absolute leaf contributions; cancellation never sets the budget.
            for b in range(nb):
                error = we[b]+gamma(count[b]-1)*major[b]
                denominator = float(c['body_subtreemass'][w % len(c['body_subtreemass']),b])
                value = weighted[b]
                if denominator != 0:
                    value = value/denominator
                    error = error/abs(denominator)
                    error += G1*(abs(value)+error)
                center[w,b] = value; ce[w,b] = error
        values['subtree_com'] = center; errors['subtree_com'] = ce
    if phase in ('inertia','composed','mass-chain'):
        out = np.empty((2,nb,10)); oe = np.empty_like(out)
        for w in range(2):
            for b in range(nb):
                r = c['ximat'][w,b].astype(float)
                re = fe('ximat')[w,b]
                inertia = c['body_inertia'][w % len(c['body_inertia']),b].astype(float)
                mass = exact(c['body_mass'][w % len(c['body_mass']),b]); root = c['root'][b]
                delta = add((c['xipos'][w,b].astype(float),fe('xipos')[w,b]),(-center[w,root],ce[w,root]))
                # Dense tensors are independent of the CUDA vec10 formula.
                tensor = r @ np.diag(inertia) @ r.T
                major = (abs(r)+re) @ np.diag(abs(inertia)) @ (abs(r)+re).T
                inherited = re @ np.diag(abs(inertia)) @ (abs(r)+re).T + abs(r) @ np.diag(abs(inertia)) @ re.T
                axes = [(delta[0][i],delta[1][i]) for i in range(3)]
                for k,(i,j) in enumerate(((0,0),(1,1),(2,2),(0,1),(0,2),(1,2))):
                    rot = tensor[i,j],inherited[i,j]+gamma(8)*major[i,j]
                    if i == j:
                        other = [a for a in range(3) if a != i]
                        shift = multiply(mass,add(multiply(axes[other[0]],axes[other[0]]),multiply(axes[other[1]],axes[other[1]])))
                    else:
                        q = multiply(multiply(mass,axes[i]),axes[j]); shift = -q[0],q[1]
                    out[w,b,k],oe[w,b,k] = add(rot,shift)
                for k in range(3):out[w,b,k+6],oe[w,b,k+6] = multiply(mass,axes[k])
                out[w,b,9],oe[w,b,9] = mass
        values['cinert'] = out; errors['cinert'] = oe
    if phase in ('axes','composed','mass-chain'):
        out = np.zeros((2,nv,6)); oe = np.zeros_like(out)
        for w in range(2):
            for j in range(nj):
                b,at,kind = int(c['jbody'][j]),int(c['dofadr'][j]),int(c['joint_type'][j]); root = c['root'][b]
                delta = add((center[w,root],ce[w,root]),(-c['xanchor'][w,j].astype(float),fe('xanchor')[w,j]))
                if kind == 0:
                    out[w,at:at+3,3:] = np.eye(3); at += 3
                if kind in (0,1):
                    for k in range(3):
                        axis = c['xmat'][w,b,:,k].astype(float),fe('xmat')[w,b,:,k]
                        out[w,at+k,:3] = axis[0]; oe[w,at+k,:3] = axis[1]
                        out[w,at+k,3:],oe[w,at+k,3:] = cross(axis,delta)
                elif kind == 2:
                    out[w,at,3:] = c['xaxis'][w,j]; oe[w,at,3:] = fe('xaxis')[w,j]
                else:
                    axis = c['xaxis'][w,j].astype(float),fe('xaxis')[w,j]
                    out[w,at,:3] = axis[0]; oe[w,at,:3] = axis[1]
                    out[w,at,3:],oe[w,at,3:] = cross(axis,delta)
        values['cdof'] = out; errors['cdof'] = oe
    if phase == 'mass-chain':
        crb = values['cinert'].copy(); cre = errors['cinert'].copy()
        major = abs(crb)+cre; count = np.ones(nb,int)
        for b in range(nb-1,0,-1):
            p = c['parent'][b]
            if p:
                crb[:,p] += crb[:,b]; cre[:,p] += cre[:,b]; major[:,p] += major[:,b]; count[p] += count[b]
        cre += gamma(count-1)[None,:,None]*major
        matrix = np.empty((2,c['nC'])); me = np.empty_like(matrix)
        for w in range(2):
            for i in range(nv):
                b = c['dof_body'][i]; tensor = spatial(crb[w,b]); te = abs(spatial(cre[w,b]))
                v,ve = values['cdof'][w,i],errors['cdof'][w,i]
                product,pe = np.zeros(6),np.zeros(6)
                for k in range(6):product,pe = add((product,pe),multiply((tensor[:,k],te[:,k]),(v[k],ve[k])))
                for at in range(c['row'][i],c['row'][i]+c['nnz'][i]):
                    j = c['column'][at]; total = exact(0.)
                    for k in range(6):total = add(total,multiply((values['cdof'][w,j,k],errors['cdof'][w,j,k]),(product[k],pe[k])))
                    if i == j:total = add(total,exact(c['dof_armature'][w % len(c['dof_armature']),i]))
                    matrix[w,at],me[w,at] = total
        values.update(crb=crb,M=matrix);errors.update(crb=cre,M=me)
    return values,errors
