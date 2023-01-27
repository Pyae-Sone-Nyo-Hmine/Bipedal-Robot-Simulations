if __name__ == '__main__':
    from ml_logger import instr
    from examples import RUN
    import jaynes
    from sac_dennis_rff.sac import train
    from sac_dennis_rff.config import Args, Actor, Critic, Agent
    from params_proto.hyper import Sweep
    import time
    from ml_logger import logger

    sweep = Sweep(RUN, Args, Actor, Critic, Agent).load("walker_sac.jsonl")
    jaynes.config('supercloud-tg', verbose=True)
    for i, kwargs in enumerate(sweep):
        # RUN.job_name = f"{{now:%H.%M.%S}}/{Args.env_name.split(':')[-1][:-3]}/{Args.seed}"
        # RUN.CUDA_VISIBLE_DEVICES = str(i)
        start = time.time()
        thunk = instr(train, **kwargs)
        jaynes.run(thunk)
        end = time.time()
        time_taken = end - start
        fps = 1 / time_taken
        logger.log(metrics={'some_val/smooth': 10, 'status': f"step ({i})"}, FPS=fps)

    jaynes.listen()
