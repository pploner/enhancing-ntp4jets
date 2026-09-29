import torch


class ReplaceLRU:
    """Replace codes unused for ``timeout`` configured clock steps.

    Forward hooks only collect code usage and replacement candidates. Calling
    :meth:`optimizer_step` advances the inactivity clock once and performs any
    required replacements. With the ``optimizer`` clock, this keeps a physical
    batch and the same effective batch assembled through gradient accumulation
    equivalent for replacement. The ``forward`` clock retains legacy behavior.
    """

    VALID_POLICIES = ["input_random", "input_kmeans", "self"]

    def __init__(self, rho=1e-4, timeout=100, clock="forward"):
        assert timeout > 1
        assert rho > 0.0
        if clock not in {"forward", "optimizer"}:
            raise ValueError(
                "Replacement clock must be 'forward' or 'optimizer', "
                f"got {clock!r}"
            )
        self.rho = rho
        self.timeout = timeout
        self.clock = clock
        self.policy = "input_random"
        self.tau = 2.0
        self._pending_usage = None
        self._pending_inputs = []
        assert self.policy in self.VALID_POLICIES

    @staticmethod
    def apply(module, rho=0.0, timeout=100, clock="forward"):
        """Register collection hooks and optimizer-step replacement state."""
        fn = ReplaceLRU(rho, timeout, clock)
        module.register_forward_hook(fn)
        module.register_buffer("_counts", timeout * torch.ones(module.num_codes))
        module._lru_replacement = fn
        return fn

    def __call__(self, module, inputs, outputs):
        """Collect usage and detached latent candidates from one microbatch."""
        if not module.training:
            return outputs

        q = outputs[1]["q"].detach().reshape(-1)
        usage = torch.bincount(q, minlength=module.num_codes).to(module._counts)
        if self._pending_usage is None:
            self._pending_usage = usage
        else:
            self._pending_usage += usage
        self._pending_inputs.append(outputs[1]["z"].detach().flatten(0, -2))
        if self.clock == "forward":
            self.optimizer_step(module)
        return outputs

    def _clear_pending(self):
        self._pending_usage = None
        self._pending_inputs = []

    @torch.no_grad()
    def optimizer_step(self, module):
        """Advance replacement once after an optimizer update."""
        if self._pending_usage is None:
            return

        usage = self._pending_usage
        inputs = self._pending_inputs
        self._clear_pending()

        module._counts -= 1
        used_indices = torch.nonzero(usage > 0, as_tuple=False).squeeze(-1)
        module._counts.index_fill_(0, used_indices, self.timeout)

        inactive_indices = torch.nonzero(
            module._counts <= 0, as_tuple=False
        ).squeeze(-1)
        num_inactive = inactive_indices.numel()
        if num_inactive == 0:
            return

        if self.policy == "self":
            probabilities = torch.zeros_like(module._counts)
            probabilities[used_indices] = usage[used_indices]
            probabilities = probabilities / probabilities.sum()
            probabilities = torch.exp(self.tau * probabilities) - 1
            selected_indices = torch.multinomial(
                probabilities, num_inactive, replacement=True
            )
            selected_values = module.codebook.weight.data[selected_indices].clone()
        else:
            z_e = torch.cat(inputs, dim=0)
            if self.policy == "input_random":
                z_e = z_e[torch.randperm(z_e.size(0), device=z_e.device)]
                repeats = num_inactive // z_e.size(0) + 1
                if repeats > 1:
                    z_e = torch.cat(repeats * [z_e])
                selected_values = z_e[:num_inactive]
            elif self.policy == "input_kmeans":
                from torchpq.clustering import KMeans

                kmeans = KMeans(
                    n_clusters=num_inactive,
                    distance="euclidean",
                    init_mode="kmeans++",
                )
                kmeans.fit(z_e.data.T.contiguous())
                selected_values = kmeans.centroids.T
            else:
                raise ValueError(f"Unknown replacement policy: {self.policy}")

        if self.rho > 0:
            norm = selected_values.norm(p=2, dim=-1, keepdim=True)
            selected_values = selected_values + self.rho * norm * torch.randn_like(
                selected_values
            )

        module.codebook.weight.data[inactive_indices] = selected_values
        module._counts.index_fill_(0, inactive_indices, self.timeout)


def lru_replacement(vq_module, rho=1e-4, timeout=100, clock="forward"):
    """Enable LRU replacement on a VQ module."""
    ReplaceLRU.apply(vq_module, rho, timeout, clock)
    return vq_module


def step_lru_replacement(module):
    """Flush accumulated usage for every replacement-enabled child module."""
    for child in module.modules():
        replacement = getattr(child, "_lru_replacement", None)
        if replacement is not None:
            replacement.optimizer_step(child)
