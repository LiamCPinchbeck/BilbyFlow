"""
bilbyflow.training.curriculum — dL-cap curriculum schedule.

Each stage caps the maximum luminosity distance (equivalently, restricts to
the louder end of the SNR distribution) and introduces fainter, more distant
signals later. Explicit curriculum_stages in the config override the single
full-range default.

The point of the curriculum is to get the embedding and flow to understand 
salient features of the waveforms before being overwhelmed by noise. Later
versions of BilbyFlow may change this dynamic in favour of SNR thresholding 
or SNR-weighted loss objectives.
"""

__all__ = ["build_curriculum_stages"]


def build_curriculum_stages(cfg, dL_full, resume_lr=None):
    stages_cfg = cfg.get("curriculum_stages", None)
    if stages_cfg:
        stages = []
        for s in stages_cfg:
            stages.append(dict(
                dL_max=float(s.get("dL_max", dL_full)),
                epochs=int(s["epochs"]),
                lr=float(s.get("lr", cfg["learning_rate"])),
                t_max=int(s.get("t_max", cfg["t_max"])),
                patience=int(s.get("patience", cfg["stop_after_epochs"])),
            ))
        if stages[-1]["dL_max"] < dL_full:
            print(f"  [curriculum] WARNING: final stage dL_max={stages[-1]['dL_max']} "
                  f"< true prior max {dL_full}.")
        return stages
    lr = float(resume_lr) if resume_lr is not None else float(cfg["learning_rate"])
    return [dict(dL_max=dL_full, epochs=int(cfg["max_num_epochs"]),
                 lr=lr, t_max=int(cfg["t_max"]),
                 patience=int(cfg["stop_after_epochs"]))]