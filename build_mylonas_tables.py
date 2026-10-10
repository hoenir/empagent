"""Precompute the empowerment tables C[x, E, b1, b2] for the Mylonas replication.

Two tables: their environment unmodified, and the same environment plus the
Part I graded actuator gate.  Cached to data/.
"""
import os

for _v in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
           'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[_v] = '1'          # one BLAS thread per worker, set before numpy

import time                                                    # noqa: E402
import numpy as np                                             # noqa: E402

from empower.mylonas import WORLDS, build_table               # noqa: E402

WORKERS = int(os.environ.get('MYL_WORKERS', 8))

if __name__ == '__main__':
    n = int(os.environ.get('MYL_N', 5))
    wname = os.environ.get('MYL_WORLD', 'theirs')
    world = WORLDS[wname]
    suffix = '' if wname == 'theirs' else '_' + wname
    for gated in (False, True):
        path = 'data/mylonas_C_n%d_%s%s.npz' % (n, 'gated' if gated else 'plain', suffix)
        if os.path.exists(path):
            print('skip (exists)', path, flush=True)
            continue
        t = time.time()
        print('building', path, flush=True)
        tab = build_table(n=n, gated=gated, world=world, workers=WORKERS, progress=100)
        np.savez_compressed(path, table=tab, n=n, gated=gated)
        print('  done %.1f min  range %.3f .. %.3f bits'
              % ((time.time() - t) / 60, tab.min(), tab.max()), flush=True)
