from lbm2.eval.runners.libero import LiberoRunner
from lbm2.eval.runners.robocasa import RoboCasaEvalRunner
from lbm2.eval.runners.lbm_eval import LBMEval


def get_eval_runner(eval_params):
    if eval_params.env == "robocasa":
        return RoboCasaEvalRunner(eval_params)
    elif eval_params.env == "libero":
        return LiberoRunner(eval_params)
    elif eval_params.env == "lbm_eval":
        return LBMEval(eval_params)
    else:
        raise ValueError(f"Invalid environment name: {eval_params.env}")
