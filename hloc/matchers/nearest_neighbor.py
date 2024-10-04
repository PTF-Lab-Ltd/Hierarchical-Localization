import torch

from ..utils.base_model import BaseModel


def find_nn(sim, top_k):
    _, ind_nn = sim.topk(top_k, dim=-1, largest=True)
    matches = ind_nn.permute(0, 2, 1).reshape(ind_nn.shape[0], -1)
    return matches, None


def mutual_check(m0, m1):
    inds0 = torch.arange(m0.shape[-1], device=m0.device)
    loop = torch.gather(m1, -1, torch.where(m0 > -1, m0, m0.new_tensor(0)))
    ok = (m0 > -1) & (inds0 == loop)
    m0_new = torch.where(ok, m0, m0.new_tensor(-1))
    return m0_new


class NearestNeighbor(BaseModel):
    default_conf = {
        "ratio_threshold": None,
        "distance_threshold": None,
        "do_mutual_check": True,
        "top_k": 1
    }
    required_inputs = ["descriptors0", "descriptors1"]

    def _init(self, conf):
        pass

    def _forward(self, data):
        if data["descriptors0"].size(-1) == 0 or data["descriptors1"].size(-1) == 0:
            matches0 = torch.full(
                data["descriptors0"].shape[:2], -1, device=data["descriptors0"].device
            )
            return {
                "matches0": matches0,
                "matching_scores0": torch.zeros_like(matches0),
            }
            
        sim = torch.einsum("bdn,bdm->bnm", data["descriptors0"], data["descriptors1"])
        matches0, scores0 = find_nn(
            sim, self.conf["top_k"]
        )

        return {
            "matches0": matches0,
            "matching_scores0": scores0,
        }
