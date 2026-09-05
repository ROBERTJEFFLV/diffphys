"""Deployable necessary excitation checks, never a Fisher/stability certificate.

Completed command/response pairs accumulate over one stationary-parameter
episode.  A collective command alone cannot authorize angular effectiveness
or inertia.  Offline mean accuracy and calibration remain separate gates.
"""
from __future__ import annotations

import torch
from identification_features import modal

INFORMATION_DIM = 49  # count + sum(x)[6] + sum(xx')[36] + response energy[4] + rise/fall[2]


def advance_information(ledger: torch.Tensor, innovation: torch.Tensor,
                        force_response: torch.Tensor, angular_response: torch.Tensor,
                        omega: torch.Tensor, available: torch.Tensor) -> torch.Tensor:
    x = torch.cat((modal(innovation), omega[:, :2] * omega[:, 2:3] / 25.0), -1)
    y = torch.cat((force_response[:, 2:3], angular_response), -1)
    signs = torch.stack((innovation.clamp_min(0).square().mean(-1),
                         innovation.clamp_max(0).square().mean(-1)), -1)
    row = torch.cat((torch.ones_like(x[:, :1]), x,
                     (x[:, :, None] * x[:, None, :]).flatten(1), y.square(), signs), -1)
    return ledger + available.reshape(-1, 1).to(row) * row


def capability_support(ledger: torch.Tensor) -> torch.Tensor:
    if ledger.ndim != 2 or ledger.shape[-1] != INFORMATION_DIM:
        raise ValueError("information ledger must be [batch,49]")
    finite = torch.isfinite(ledger).all(-1)
    safe = torch.where(finite[:, None], ledger, torch.zeros_like(ledger)).double()
    count, total = safe[:, :1].clamp_min(1), safe[:, 1:7]
    gram = safe[:, 7:43].reshape(-1, 6, 6) - total[:, :, None] * total[:, None, :] / count[:, :, None]
    gram = .5 * (gram + gram.transpose(-1, -2))
    # Project each motor mode off the other modes and inertial coupling terms.
    # Regularization is only numerical; it must not manufacture excitation.
    energies = []
    for axis in range(4):
        others = [i for i in range(6) if i != axis]
        g = gram[:, others][:, :, others]
        cross = gram[:, axis, others]
        projection = (cross[:, None] @ torch.linalg.pinv(g, rtol=1e-7) @ cross[:, :, None]).flatten()
        energies.append((gram[:, axis, axis] - projection).clamp_min(0))
    excited = torch.stack(energies, -1) >= 1e-3
    response = safe[:, 43:47] >= 1e-6
    collective = excited[:, 0] & response[:, 0]
    angular = excited[:, 1:4] & response[:, 1:4]
    # Gyroscopic products must add information beyond motor-mode excitation;
    # raw nonzero omega products alone can be completely collinear with it.
    coupling = gram[:, 4:, :4]
    conditional = gram[:, 4:, 4:] - coupling @ torch.linalg.pinv(gram[:, :4, :4], rtol=1e-7) @ coupling.transpose(-1, -2)
    inertial = conditional.diagonal(dim1=-2, dim2=-1).sum(-1) >= 1e-4
    supported = torch.stack((collective, angular[:, :2].all(-1),
                             angular.all(-1), angular[:, :2].all(-1) & inertial,
                             collective & (safe[:, 47] >= 1e-3),
                             collective & (safe[:, 48] >= 1e-3)), -1)
    return supported & finite[:, None] & (count >= 12)
