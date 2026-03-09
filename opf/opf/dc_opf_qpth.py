# opf/dc_opf_qpth.py
import numpy as np
import torch

class SoftDCOPF_QPTH:
    """
    soft DC-OPF (always feasible):
      variables: [pg(ng), theta(nonref)(nb-1), s(nb), r(nb), u+(nl), u-(nl)]
      - s: load shedding (>=0)
      - r: spill/curtailment as extra demand (>=0) to absorb negative pd / surplus gen
      - u+/u-: line limit slacks (>=0)
    solved by qpth, supports BP wrt pd
    """
    def __init__(
        self,
        baseMVA=100.0,
        eps_q=1e-6,
        Ms=1e3,
        Mr=5e2,
        Mu=1e2,
        dtype=torch.float64,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.dtype = dtype

        # --- imports (HPC friendly: no auto-install) ---
        try:
            from qpth.qp import QPFunction
        except Exception as e:
            raise ImportError(
                "Missing dependency: qpth. Please install it in your env: `pip install qpth`"
            ) from e

        try:
            # pypower needs scipy
            from pypower.api import case30, makeBdc
        except Exception as e:
            raise ImportError(
                "Missing dependency: pypower (and likely scipy). Please install: `pip install pypower scipy`"
            ) from e

        self.QPFunction = QPFunction

        # --- load IEEE30 ---
        ppc = case30()
        ppc["baseMVA"] = float(baseMVA)

        baseMVA = float(ppc["baseMVA"])
        bus = ppc["bus"]
        gen = ppc["gen"]
        branch = ppc["branch"]
        gencost = ppc["gencost"]

        self.baseMVA = baseMVA
        self.bus = bus
        self.gen = gen
        self.branch = branch
        self.gencost = gencost

        nb = bus.shape[0]
        ng = gen.shape[0]
        nl = branch.shape[0]
        self.nb, self.ng, self.nl = nb, ng, nl

        # ref bus (BUS_TYPE == 3), bus[:,1] is BUS_TYPE in MATPOWER/PYPOWER
        ref_candidates = np.where(bus[:, 1] == 3)[0]
        ref = int(ref_candidates[0]) if len(ref_candidates) > 0 else 0
        self.ref = ref
        self.nonref = np.array([i for i in range(nb) if i != ref], dtype=np.int64)
        self.theta_dim = nb - 1

        # build B matrices (may be scipy sparse)
        # --- convert external bus numbering (often 1..nb) to internal 0..nb-1 ---
        bus = bus.copy()
        branch = branch.copy()
        gen = gen.copy()

        nb = bus.shape[0]

        # external bus numbers as ints
        ext_bus = bus[:, 0].astype(int)  # e.g. 1..30
        # map external -> internal index 0..nb-1
        ext2int = {int(b): i for i, b in enumerate(ext_bus)}

        # overwrite bus numbers to internal 0..nb-1
        bus[:, 0] = np.arange(nb)

        # branch f/t bus (cols 0,1) -> internal
        branch[:, 0] = np.array([ext2int[int(b)] for b in branch[:, 0].astype(int)], dtype=branch.dtype)
        branch[:, 1] = np.array([ext2int[int(b)] for b in branch[:, 1].astype(int)], dtype=branch.dtype)

        # gen bus (col 0) -> internal
        gen[:, 0] = np.array([ext2int[int(b)] for b in gen[:, 0].astype(int)], dtype=gen.dtype)

        # now safe to call makeBdc
        Bbus, Bf, Pbusinj, Pfinj = makeBdc(baseMVA, bus, branch)

        def to_dense(mat):
            if hasattr(mat, "toarray"):
                return mat.toarray()
            return np.asarray(mat)

        Bbus = to_dense(Bbus).astype(np.float64)     # (nb, nb)
        Bf = to_dense(Bf).astype(np.float64)         # (nl, nb)
        Pbusinj = np.asarray(Pbusinj, dtype=np.float64).reshape(-1)  # (nb,)
        Pfinj = np.asarray(Pfinj, dtype=np.float64).reshape(-1)      # (nl,)

        Bbus_nr = Bbus[:, self.nonref]  # (nb, nb-1)
        Bf_nr = Bf[:, self.nonref]      # (nl, nb-1)

        # map bus number -> row index (robust)
        bus_numbers = bus[:, 0].astype(int)
        bus_lookup = {int(num): i for i, num in enumerate(bus_numbers)}

        # Cg mapping (nb x ng): gen at which bus
        Cg = np.zeros((nb, ng), dtype=np.float64)
        for g in range(ng):
            bus_no = int(gen[g, 0])
            bidx = bus_lookup[bus_no]
            Cg[bidx, g] = 1.0

        # generator bounds (MW -> pu)
        # gen cols: PMAX=8, PMIN=9
        Pmax = gen[:, 8].astype(np.float64) / baseMVA
        Pmin = gen[:, 9].astype(np.float64) / baseMVA

        # branch limits (rateA=5), MW -> pu
        rateA = branch[:, 5].astype(np.float64)
        Fmax = np.where(rateA > 0, rateA, 1e4) / baseMVA
        self.Fmax = Fmax

        # costs: gencost polynomial, handle ncost 1/2/3
        c2 = np.zeros(ng, dtype=np.float64)
        c1 = np.zeros(ng, dtype=np.float64)
        for i in range(ng):
            row = gencost[i]
            ncost = int(row[3])
            coeffs = row[4:4 + ncost]  # highest order first
            if ncost == 3:
                c2[i], c1[i] = coeffs[0], coeffs[1]
            elif ncost == 2:
                c2[i], c1[i] = 0.0, coeffs[0]
            elif ncost == 1:
                c2[i], c1[i] = 0.0, 0.0
            else:
                c2[i], c1[i] = 0.0, 0.0

        # variable dims
        i_pg = 0
        i_th = i_pg + ng
        i_s  = i_th + self.theta_dim
        i_r  = i_s  + nb
        i_up = i_r  + nb
        i_un = i_up + nl
        n = ng + self.theta_dim + nb + nb + nl + nl
        self.n = n

        self._idx = dict(i_pg=i_pg, i_th=i_th, i_s=i_s, i_r=i_r, i_up=i_up, i_un=i_un)

        # Q, p (constant base)
        Q = np.zeros((n, n), dtype=np.float64)
        p = np.zeros((n,), dtype=np.float64)

        # quadratic on pg, enforce PD by eps_q
        for g in range(ng):
            Q[i_pg + g, i_pg + g] = 2.0 * c2[g] * (baseMVA ** 2) + eps_q
            p[i_pg + g] = c1[g] * baseMVA

        # small regularization on all other vars
        for k in range(i_th, n):
            Q[k, k] += eps_q

        # linear penalties on slacks
        p[i_s:i_s + nb] = Ms
        p[i_r:i_r + nb] = Mr
        p[i_up:i_up + nl] = Mu
        p[i_un:i_un + nl] = Mu

        self.Q_base = torch.tensor(Q, dtype=dtype, device=self.device)
        self.p_base = torch.tensor(p, dtype=dtype, device=self.device)

        # Equality: Cg*pg - Bbus_nr*theta + s - r = pd_pu + Pbusinj
        Aeq = np.zeros((nb, n), dtype=np.float64)
        Aeq[:, i_pg:i_pg + ng] = Cg
        Aeq[:, i_th:i_th + self.theta_dim] = -Bbus_nr
        Aeq[:, i_s:i_s + nb] = np.eye(nb)
        Aeq[:, i_r:i_r + nb] = -np.eye(nb)

        self.Aeq = torch.tensor(Aeq, dtype=dtype, device=self.device)
        self.Pbusinj = torch.tensor(Pbusinj, dtype=dtype, device=self.device)

        # Inequalities Gx <= h
        G_list, h_list = [], []

        # pg <= Pmax
        G = np.zeros((ng, n), dtype=np.float64)
        G[:, i_pg:i_pg + ng] = np.eye(ng)
        G_list.append(G); h_list.append(Pmax)

        # -pg <= -Pmin
        G = np.zeros((ng, n), dtype=np.float64)
        G[:, i_pg:i_pg + ng] = -np.eye(ng)
        G_list.append(G); h_list.append(-Pmin)

        # line upper: Bf_nr*theta - u+ <= Fmax - Pfinj
        G = np.zeros((nl, n), dtype=np.float64)
        G[:, i_th:i_th + self.theta_dim] = Bf_nr
        G[:, i_up:i_up + nl] = -np.eye(nl)
        G_list.append(G); h_list.append(Fmax - Pfinj)

        # line lower: -Bf_nr*theta - u- <= Fmax + Pfinj
        G = np.zeros((nl, n), dtype=np.float64)
        G[:, i_th:i_th + self.theta_dim] = -Bf_nr
        G[:, i_un:i_un + nl] = -np.eye(nl)
        G_list.append(G); h_list.append(Fmax + Pfinj)

        # nonnegativity: -s <= 0, -r <= 0, -u+<=0, -u-<=0
        for (start, m) in [(i_s, nb), (i_r, nb), (i_up, nl), (i_un, nl)]:
            G = np.zeros((m, n), dtype=np.float64)
            G[:, start:start + m] = -np.eye(m)
            G_list.append(G); h_list.append(np.zeros(m, dtype=np.float64))

        G = np.vstack(G_list)
        h = np.concatenate(h_list)

        self.G_base = torch.tensor(G, dtype=dtype, device=self.device)
        self.h_base = torch.tensor(h, dtype=dtype, device=self.device)

    def get_base_pd_mw(self):
        # bus Pd column is 2 (MW)
        return self.bus[:, 2].astype(np.float64)

    def build_b(self, pd_mw: torch.Tensor):
        # pd_mw: (B, nb) MW -> pu
        pd_pu = pd_mw.to(self.dtype) / self.baseMVA
        return pd_pu + self.Pbusinj.unsqueeze(0)

    def solve(self, pd_mw: torch.Tensor, qp_pg_rho: float = 0.0, qp_pg_v: torch.Tensor = None, gen_fix: torch.Tensor = None):
        B = pd_mw.shape[0]
        Q = self.Q_base.unsqueeze(0).expand(B, self.n, self.n).clone()
        p = self.p_base.unsqueeze(0).expand(B, self.n).clone()

        # optional PHA penalty on pg: (rho/2)||pg - v||^2
        if qp_pg_rho > 0.0 and qp_pg_v is not None:
            i_pg = self._idx["i_pg"]
            ng = self.ng
            rho = float(qp_pg_rho)
            eye = torch.eye(ng, dtype=self.dtype, device=self.device).unsqueeze(0)
            Q[:, i_pg:i_pg + ng, i_pg:i_pg + ng] += rho * eye
            p[:, i_pg:i_pg + ng] += (-rho) * qp_pg_v.to(self.dtype)

        b = self.build_b(pd_mw)

        G = self.G_base.unsqueeze(0).expand(B, self.G_base.shape[0], self.n)
        h = self.h_base.unsqueeze(0).expand(B, self.h_base.shape[0]).clone()

        # if fix pg, overwrite bounds
        if gen_fix is not None:
            gen_fix = gen_fix.to(self.dtype)
            ng = self.ng
            h[:, 0:ng] = gen_fix
            h[:, ng:2 * ng] = -gen_fix

        Aeq = self.Aeq.unsqueeze(0).expand(B, self.nb, self.n)

        x = self.QPFunction(verbose=-1)(Q, p, G, h, Aeq, b)

        xQx = torch.einsum("bi,bij,bj->b", x, Q, x)
        px = torch.einsum("bi,bi->b", p, x)
        obj = 0.5 * xQx + px
        return x, obj