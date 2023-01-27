from pathlib import Path

from params_proto.hyper import Sweep

from drq.config import Args, Agent
from sysid import RUN

with Sweep(RUN, Args, Agent) as sweep:
    Args.from_pixels = False
    Args.save_video = False
    Args.save_final_replay_buffer = True
    Args.env_name = 'lcs:Paramcartpole-swingup-v1'
    Args.feature_dim = 50
    Args.train_frames = 1_000_000
    Args.checkpoint_frequency = 30_000
    Args.replay_buffer_size = 1_000_000
    Agent.batch_size = 256
    Agent.lr = 1e-4

    with sweep.product:
        Args.env_reset_kwargs = [{'cart_mass': 1.0, 'pole_mass': 0.1},
                                 {'cart_mass': 3.0, 'pole_mass': 0.1},
                                 {'cart_mass': 1.0, 'pole_mass': 1.5},
                                 {'cart_mass': 3.0, 'pole_mass': 0.7},
                                 {'cart_mass': 1.0, 'pole_mass': 0.7}]
        Args.seed = [100, 200, 300, 400, 500]

    @sweep.each
    def tail(RUN, Args, Agent):
        RUN.job_name = (f"Cart{int(Args.env_reset_kwargs['cart_mass'] * 10):02d}"
                        f"Pole{int(Args.env_reset_kwargs['pole_mass'] * 10):02d}"
                        f"/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")
