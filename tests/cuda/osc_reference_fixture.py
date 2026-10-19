"""Write same-input fixtures using the installed robosuite CPU implementation."""
import argparse
from pathlib import Path
import struct
import numpy as np
from robosuite.utils.control_utils import opspace_matrices, nullspace_torques, orientation_error


def expected(v):
    result = {}
    ranks = []
    for w in range(2):
        m, j = v['pm'][w].reshape(7, 7), v['pj'][w].reshape(6, 7)
        mi = np.linalg.inv(m)
        a = j @ mi @ j.T
        lf, lp, lr, n = opspace_matrices(m, j, j[:3], j[3:])
        pos = v['goal_pos'][w] - v['pp'][w]
        ori = orientation_error(v['goal_rot'][w].reshape(3, 3), v['pr'][w].reshape(3, 3))
        desired = v['gains'][w, :6] * np.r_[pos, ori] - v['gains'][w, 6:] * v['pe'][w]
        nt = nullspace_torques(m, n, v['q0'][w], v['pq'][w], v['pv'][w])
        tau = j.T @ np.r_[lp @ desired[:3], lr @ desired[3:]] + v['bias'][w] + nt
        limits = v['range'][w].reshape(9, 2)
        clipped = np.clip(tau, limits[:7, 0], limits[:7, 1])
        values = dict(mass_inv=mi, A_full=a, A_pos=a[:3, :3], A_ori=a[3:, 3:],
                      lambda_full=lf, lambda_pos=lp, lambda_ori=lr, nullspace_matrix=n,
                      position_error=pos, orientation_error=ori, nullspace_torque=nt,
                      raw_tau=tau, clipped=clipped, ctrl=np.r_[clipped, limits[7:].mean(axis=1)],
                      goal_pos=v['goal_pos'][w], goal_rot=v['goal_rot'][w])
        for name, value in values.items():
            result.setdefault(name, []).append(np.asarray(value).reshape(-1))
        ranks.append([int(np.sum((s := np.linalg.svd(x, compute_uv=False)) > 1e-15 * s.max()))
                      for x in (a, a[:3, :3], a[3:, 3:])])
    return result, ranks


def main(path):
    rng = np.random.default_rng(327)
    with path.open('wb') as f:
        f.write(struct.pack('<QII', 0x4f53435f4e325631, 4, 2))
        for case in range(4):
            m = rng.normal(size=(2, 7, 7))
            m = m @ m.transpose(0, 2, 1) + np.eye(7) * 2
            j = rng.normal(size=(2, 6, 7))
            if case == 1:
                j[0, 4:] = 0
                j[1, 5:] = 0
            elif case == 2:
                j *= 1e-3
            elif case == 3:
                j[:] = 0
            rot = np.tile(np.eye(3).reshape(1, 9), (2, 1))
            goals = rot.copy()
            c, s = np.cos(.07), np.sin(.07)
            goals[:] = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).reshape(1, 9)
            v = dict(pm=m.reshape(2, 49), pj=j.reshape(2, 42), pq=rng.normal(size=(2, 7)) * .1,
                     pv=rng.normal(size=(2, 7)) * .05, bias=rng.normal(size=(2, 7)),
                     q0=np.zeros((2, 7)), pp=rng.normal(size=(2, 3)) * .05, pr=rot,
                     pe=rng.normal(size=(2, 6)) * .02, goal_pos=rng.normal(size=(2, 3)) * .05,
                     goal_rot=goals, gains=np.tile(np.r_[np.full(6, 150.), np.full(6, 2*np.sqrt(150.))], (2, 1)),
                     range=np.tile(np.array([-87., 87.]), (2, 9)))
            for value in v.values():
                f.write(np.asarray(value, dtype='<f8').tobytes())
            for step in range(3):
                if step == 2:
                    v['pm'][1, 0] += .125
                results, ranks = expected(v)
                for value in results.values():
                    f.write(np.asarray(value, dtype='<f8').tobytes())
                f.write(np.asarray(ranks, dtype='<i4').tobytes())
    print(f'CPU OSC fixture: 4 cases, 2 worlds, 3 cache transitions: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    main(parser.parse_args().output)
