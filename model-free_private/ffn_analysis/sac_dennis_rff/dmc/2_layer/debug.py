from pathlib import Path

from params_proto.hyper import Sweep

from sac_dennis_rff.config import Args, Actor, Critic, Agent
from ffn_analysis import RUN

with Sweep(RUN, Args, Actor, Critic, Agent) as sweep:
    Args.dmc = True
    Args.train_frames = 1_000_000
    Args.env_name = 'dmc:Quadruped-run-v1'
    Args.checkpoint_root = "gs://your-gs-bucket"
    Args.save_final_replay_buffer = True

    RUN.prefix = "{project}/{project}/{file_stem}/{job_name}"
    Args.seed = 100
    Agent.use_rff = True
    Agent.scale = 5
    Agent.actor_fourier_features = 3120
    Agent.critic_fourier_features = 3600

    # with sweep.product:
    #     Args.seed = [100, 200, 300, 400, 500]


@sweep.each
def tail(RUN, Args, Actor, Critic, Agent, *_):
    RUN.prefix, RUN.job_name, _ = RUN(script_path=__file__,
                                      job_name=f"rff/{Args.env_name.split(':')[-1][:-3]}/{Args.seed}")


sweep.save(f"{Path(__file__).stem}.jsonl")