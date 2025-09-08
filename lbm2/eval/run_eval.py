import argparse
import imageio
from tqdm import tqdm
from lbm2.eval.runners import get_eval_runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument("--num_steps", type=int, default=100)
    parser.add_argument("--video_path", type=str, default="./test.mp4")
    return parser.parse_args()

def run_eval(args):
    eval_runner = get_eval_runner(args)
    eval_runner.load_env(args.task)
    eval_runner.load_model(args.model_path)
    
    video_writer = None
    if args.video_path is not None:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    num_success_rollouts = 0
    for rollout_i in tqdm(range(args.num_rollouts)):
        obs = eval_runner.env_reset()
        for step_i in range(args.num_steps):
            obs_extracted = eval_runner.extract_from_obs(obs)
            action = eval_runner.model.get_action(obs_extracted['images'], obs_extracted['text'])

            obs = eval_runner.env_step(action)
            obs = eval_runner.get_obs_tensor(obs)
            
            if video_writer is not None:
                video_writer.append_data(eval_runner.get_current_image())

            if eval_runner.check_success():
                num_success_rollouts += 1
                break

    if args.video_path is not None:
        video_writer.close()
        print(f"Saved video of rollouts to {args.video_path}")

    print("Task accuracy: ", num_success_rollouts / args.num_rollouts)
    eval_runner.env_close()
    return


def main():
    args = parse_args()
    run_eval(args)


if __name__ == "__main__":
    main()