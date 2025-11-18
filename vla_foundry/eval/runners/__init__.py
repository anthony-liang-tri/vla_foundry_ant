def get_eval_runner(eval_params):
    if eval_params.env == "robocasa":
        from vla_foundry.eval.runners.robocasa import RoboCasaEvalRunner

        return RoboCasaEvalRunner(eval_params)
    elif "libero" in eval_params.env:
        # "libero_spatial", "libero_object", "libero_goal", "libero_10", "libero_90"
        from vla_foundry.eval.runners.libero import LiberoRunner

        return LiberoRunner(eval_params)
    elif eval_params.env == "lbm_eval":
        from vla_foundry.eval.runners.lbm_eval import LBMEval

        return LBMEval(eval_params)
    elif eval_params.env == "robosuite":
        from vla_foundry.eval.runners.robosuite import RoboSuiteEvalRunner

        return RoboSuiteEvalRunner(eval_params)
    else:
        raise ValueError(f"Invalid environment name: {eval_params.env}")
