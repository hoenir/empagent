"""Choose the patch-depletion world using objective-free reference policies only.

No empowerment agent is run here.  The question is purely about the world:
for which food dynamics does SWITCHING between sources pay?

  parker    walk to the nearest source and stand on it forever
  switcher  go to (or stay on) the source currently believed more likely full;
            so it leaves a source once it has seen it empty

Rule, fixed before any result was seen:
  in the gated world, switcher lifespan >= 2 x parker lifespan, and the
  switcher itself is viable (mean lifespan >= 1500 of the 3000-step cap).
  Among passing worlds, take the largest switcher/parker ratio.
Only lam, mu, rho vary; grid, energy cap and meal size are theirs.
"""

import itertools
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from empower.mylonas import THEIRS, Parker, Switcher, World, simulate


N_EP = 30
MAXSTEPS = 3000


def evaluate(args):
    lam, mu, rho, gated = args
    w = World(lam, mu, rho, name='cand')
    out = {}
    for name, pol in (('parker', Parker()), ('switcher', Switcher())):
        life = [len(simulate(pol, gated=gated, world=w, seed=k, max_steps=MAXSTEPS)['x'])
                for k in range(N_EP)]
        out[name] = float(np.mean(life))
    return (lam, mu, rho, gated), out


if __name__ == '__main__':
    # Grid 1 (lam 0.95-1.0, mu 0.02-0.1, rho 0.2/0.5) produced no passing world:
    # refill was too slow for anything to survive.  Grid 2 extends toward faster
    # refill and harder depletion, under the same rule.  Results of grid 1 are
    # kept in data/patch_world_selection_grid1.json.
    lams = [0.99, 1.0]
    mus = [0.1, 0.15, 0.2, 0.3]
    rhos = [0.0, 0.2, 0.5]
    jobs = [(l, m, r, g) for l, m, r in itertools.product(lams, mus, rhos)
            for g in (False, True)]
    jobs += [(THEIRS.lam, THEIRS.mu, THEIRS.rho, g) for g in (False, True)]

    with ProcessPoolExecutor(max_workers=12) as ex:
        res = dict(ex.map(evaluate, jobs))

    rows = []
    for (l, m, r) in sorted({k[:3] for k in res}):
        pl, sw = res[(l, m, r, False)], res[(l, m, r, True)]
        rows.append(dict(lam=l, mu=m, rho=r,
                         plain_parker=pl['parker'], plain_switcher=pl['switcher'],
                         gated_parker=sw['parker'], gated_switcher=sw['switcher'],
                         gated_ratio=sw['switcher'] / max(sw['parker'], 1.0),
                         plain_ratio=pl['switcher'] / max(pl['parker'], 1.0)))

    print('%5s %5s %4s | %8s %8s %5s | %8s %8s %5s  %s'
          % ('lam', 'mu', 'rho', 'parker', 'switch', 'ratio',
             'g-parker', 'g-switch', 'ratio', 'pass'))
    passing = []
    for r in rows:
        ok = (r['gated_ratio'] >= 2.0 and r['gated_switcher'] >= 1500)
        theirs = (r['lam'], r['mu'], r['rho']) == (THEIRS.lam, THEIRS.mu, THEIRS.rho)
        if ok and not theirs:
            passing.append(r)
        print('%5.2f %5.2f %4.1f | %8.0f %8.0f %5.2f | %8.0f %8.0f %5.2f  %s%s'
              % (r['lam'], r['mu'], r['rho'], r['plain_parker'], r['plain_switcher'],
                 r['plain_ratio'], r['gated_parker'], r['gated_switcher'],
                 r['gated_ratio'], 'PASS' if ok else '', '   <- theirs' if theirs else ''))

    best = max(passing, key=lambda r: r['gated_ratio']) if passing else None
    print('\nselected:', best)
    os.makedirs('data', exist_ok=True)
    with open('data/patch_world_selection.json', 'w') as fh:
        json.dump(dict(rule='gated switcher >= 2x gated parker and >= 1500 steps; '
                            'max ratio among passing', rows=rows, selected=best),
                  fh, indent=1)
