# opf/pha.py
import torch

@torch.no_grad()
def pha_solve_pg(opf_solver, pd_scenarios_mw: torch.Tensor, rho: float = 1.0, max_iter: int = 50, tol: float = 1e-3,
                 batch_chunk: int = 128):
    """
    PHA for 2-stage style with non-anticipativity on pg:
      each scenario has its own (pg^s, theta^s, slacks^s)
      enforce consensus pg_bar
    return:
      pg_bar: (ng,)
    """
    device = pd_scenarios_mw.device
    dtype = opf_solver.dtype
    S = pd_scenarios_mw.shape[0]
    ng = opf_solver.ng

    # init: solve each scenario independently
    pg = torch.zeros((S, ng), dtype=dtype, device=device)
    u = torch.zeros((S, ng), dtype=dtype, device=device)

    # solve in chunks to avoid memory explosion
    def solve_chunk(pd_chunk, qp_v_chunk):
        x, _ = opf_solver.solve(pd_chunk, qp_pg_rho=rho, qp_pg_v=qp_v_chunk)
        i_pg = opf_solver._idx["i_pg"]
        return x[:, i_pg:i_pg+ng]

    # first pass: rho=0
    for st in range(0, S, batch_chunk):
        ed = min(S, st+batch_chunk)
        x, _ = opf_solver.solve(pd_scenarios_mw[st:ed], qp_pg_rho=0.0, qp_pg_v=None)
        i_pg = opf_solver._idx["i_pg"]
        pg[st:ed] = x[:, i_pg:i_pg+ng]

    pg_bar = pg.mean(dim=0)

    for it in range(max_iter):
        # scenario subproblems
        for st in range(0, S, batch_chunk):
            ed = min(S, st+batch_chunk)
            v = (pg_bar - u[st:ed])  # v = pg_bar - u
            pg[st:ed] = solve_chunk(pd_scenarios_mw[st:ed], v)

        pg_bar_new = (pg + u).mean(dim=0)
        # update duals
        u = u + (pg - pg_bar_new.unsqueeze(0))

        # stop check
        gap = (pg - pg_bar_new.unsqueeze(0)).norm(dim=1).max().item()
        pg_bar = pg_bar_new
        if gap < tol:
            break

    return pg_bar.to(dtype)