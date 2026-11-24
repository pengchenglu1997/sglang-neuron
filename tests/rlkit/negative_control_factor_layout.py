"""Negative control: the layout check must FAIL when the factors are wrong. Three deliberate mistakes."""
import torch

from factor_builders import QWEN3_0_6B as D, build_factors, engine_loaders, live_delta
from test_factor_layout import make_pairs, SCALE

tp = 4
L = engine_loaders(D, tp)
res = {}
for kind in ("qkv", "gate_up", "o", "down"):
    pairs = make_pairs(kind, D, seed=1)
    ok_right, ok_wrong_rank, ok_unscaled, ok_noblock = 0, 0, 0, 0
    for rank in range(tp):
        a, bt = build_factors(kind, pairs, SCALE, rank, tp, D, L)
        ref = live_delta(kind, pairs, SCALE, rank, L[kind])
        ok_right += torch.equal(a.t() @ bt, ref)
        a2, bt2 = build_factors(kind, pairs, SCALE, (rank + 1) % tp, tp, D, L)          # wrong rank
        ok_wrong_rank += torch.equal(a2.t() @ bt2, ref)
        ok_unscaled += torch.equal(a.t() @ (bt / SCALE), ref)                            # scale forgotten
        if kind in ("qkv", "gate_up"):
            r = pairs[0][1].shape[0]
            bt3 = bt.clone()
            for i in range(1, len(pairs)):                                                # drop the zero blocks' structure
                bt3[:r] += bt[i * r:(i + 1) * r]
            ok_noblock += torch.equal(a.t() @ bt3, ref)
    res[kind] = (ok_right, ok_wrong_rank, ok_unscaled, ok_noblock)
    print(f"{kind:8s} ranks equal: correct={ok_right}/4  wrong-rank={ok_wrong_rank}/4  scale-forgotten={ok_unscaled}/4"
          f"  segments-summed={ok_noblock}/4")
assert all(v[0] == 4 and v[1] == 0 and v[2] == 0 for v in res.values())
print("negative control OK: correct factors pass on all ranks, each deliberate mistake fails on all ranks")
