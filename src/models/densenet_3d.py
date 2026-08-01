"""
DenseNet121-3D for 32³ nodule classification.

Modifications vs. stock MONAI DenseNet121:
  - Stem conv0: 7×7×7 stride-2 → 3×3×3 stride-1 padding-1
  - Stem pool0: MaxPool stride-2 → Identity
  - Everything else (dense blocks, transitions, growth rate) unchanged.

Rationale: stock stem collapses 32³ → 8³ before the first dense block and
→ 1³ after three transitions, losing all spatial information. The modified
stem preserves 32³ through the stem, then halves at each of the three
transitions: 32 → 16 → 8 → 4. Feature map never drops below 4³.

Shape trace: call trace_feature_maps(model, device) to print spatial dims
at each stage. Run on the lab machine to verify before training.
"""

from typing import Optional

import torch
import torch.nn as nn

try:
    from monai.networks.nets import DenseNet121
except ImportError as e:
    raise ImportError(
        "MONAI is required for DenseNet121-3D. "
        "Install with: pip install monai"
    ) from e

from src.utils.paths import find_repo_root, get_data_dir


# ── Model builder ─────────────────────────────────────────────────────────────

def build_densenet(cfg: dict, radiomics_dim: int = 0) -> nn.Module:
    """
    Build the modified DenseNet121-3D.

    Args:
        cfg:          Full config dict (from densenet_config.yaml).
        radiomics_dim: If > 0 AND cfg['use_radiomics'] is True, returns
                       DenseNet121WithRadiomics instead (Phase 8).
    Returns:
        nn.Module ready for training (on CPU; caller moves to device).
    """
    model_cfg = cfg.get("model", {})
    in_ch  = model_cfg.get("in_channels", 1)
    out_ch = model_cfg.get("out_channels", 2)

    model = DenseNet121(
        spatial_dims=3,
        in_channels=in_ch,
        out_channels=out_ch,
        pretrained=False,
    )

    # ── Replace stem ──────────────────────────────────────────────────────────
    # Original: features.conv0 = Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    #           features.pool0 = MaxPool3d(kernel_size=3, stride=2, padding=1)
    # New:      3×3×3 stride-1 conv (preserves spatial dims)
    #           Identity (no pooling)
    orig_conv0 = model.features.conv0
    model.features.conv0 = nn.Conv3d(
        in_channels=in_ch,
        out_channels=orig_conv0.out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )
    nn.init.kaiming_normal_(model.features.conv0.weight, mode="fan_out", nonlinearity="relu")
    model.features.pool0 = nn.Identity()
    # ──────────────────────────────────────────────────────────────────────────

    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"[model] DenseNet121-3D (modified stem) | {n_params:.2f}M parameters")

    # Radiomics fusion (Phase 8, disabled by default)
    use_rad = cfg.get("use_radiomics", False)
    if use_rad and radiomics_dim > 0:
        model = DenseNet121WithRadiomics(model, radiomics_dim, out_ch)
        print(f"[model] Radiomics fusion head enabled ({radiomics_dim} features)")

    return model


# ── Shape tracer ──────────────────────────────────────────────────────────────

def trace_feature_maps(model: nn.Module, device: torch.device) -> None:
    """
    Print spatial dimensions after each dense block and transition.
    Also prints peak VRAM usage for the dummy forward pass.

    Run on the lab machine before training to verify spatial dims.
    """
    handles = []
    stage_names = []

    # Identify the dense blocks and transitions in model.features
    # MONAI DenseNet names them: denseblock1, transition1, denseblock2, ...
    for name, module in model.features.named_children():
        if "denseblock" in name or "transition" in name or name in ("conv0", "norm0", "relu0", "pool0"):
            stage_names.append(name)

    print("[shape trace] Running dummy forward pass (2, 1, 32, 32, 32) ...")
    model.eval().to(device)
    hooks_output = {}

    def make_hook(n):
        def hook(mod, inp, out):
            if hasattr(out, "shape"):
                hooks_output[n] = tuple(out.shape)
        return hook

    for name, module in model.features.named_children():
        h = module.register_forward_hook(make_hook(name))
        handles.append(h)

    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        dummy = torch.zeros(2, 1, 32, 32, 32, device=device)
        _ = model(dummy)
        del dummy

    for h in handles:
        h.remove()

    print(f"  {'stage':<20} shape")
    print(f"  {'-'*20} -----")
    for stage in stage_names:
        if stage in hooks_output:
            s = hooks_output[stage]
            print(f"  {stage:<20} {s}")

    if device.type == "cuda":
        peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
        print(f"\n  Peak VRAM (batch=2): {peak_mb:.1f} MB")

    model.train()
    print("[shape trace] done")


# ── Phase 8: Radiomics fusion head ────────────────────────────────────────────

class DenseNet121WithRadiomics(nn.Module):
    """
    DenseNet121 + handcrafted radiomics features concatenated before classifier.

    The DenseNet's final FC layer is replaced by:
        global-avg-pool → cat(CNN_features, standardized_radiomics) → Linear

    The radiomics scaler MUST be fit on train folds only (never on the full
    dataset). This is enforced in scripts/extract_radiomics.py and in the
    CV training loop via a per-fold StandardScaler fitted on train_df only.
    """

    def __init__(self, base_model: nn.Module, radiomics_dim: int, out_channels: int):
        super().__init__()

        # Detach the classifier from base model
        # MONAI DenseNet final layer is model.class_layers.out
        cnn_feature_dim = base_model.class_layers.out.in_features

        # Remove the final linear from base_model; keep features + pooling
        self.features       = base_model.features
        self.class_norm     = base_model.class_layers.relu  # BN+ReLU before pool
        self.global_pool    = nn.AdaptiveAvgPool3d(1)

        # Fusion classifier
        fused_dim = cnn_feature_dim + radiomics_dim
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, out_channels),
        )

        self.radiomics_dim = radiomics_dim

    def forward(self, x: torch.Tensor, radiomics: Optional[torch.Tensor] = None):
        feat = self.features(x)
        feat = self.class_norm(feat)
        feat = self.global_pool(feat).flatten(1)

        if radiomics is not None:
            feat = torch.cat([feat, radiomics], dim=1)
        elif self.radiomics_dim > 0:
            raise ValueError(
                "Model expects radiomics features but none were provided. "
                "Pass radiomics=None or disable use_radiomics in config."
            )

        return self.classifier(feat)
