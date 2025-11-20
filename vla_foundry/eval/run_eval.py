import argparse

import imageio
from tqdm import tqdm

from vla_foundry.eval.runners import get_eval_runner


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--video_path", type=str, default="./test.mp4")

    parser.add_argument("--num_rollouts", type=int, default=3)
    parser.add_argument(
        "--num_steps",
        type=int,
        default=50,
        help=(
            "Number of trajectories to generate per rollout. This value (paired with action_window) "
            "implicitly controls the time the robot has to complete an evaluation task."
        ),
    )
    parser.add_argument("--action_window", type=int, default=8, help="Number of actions to take per generation step")
    parser.add_argument("--image_names", type=str, nargs="+", default="")
    return parser.parse_args()


def run_eval(args):
    eval_runner = get_eval_runner(args)
    eval_runner.load_env(args.env, args.task)
    eval_runner.load_model(args.model_path)

    video_writer = None
    if args.video_path is not None:
        video_writer = imageio.get_writer(args.video_path, fps=20)

    num_success_rollouts = 0
    for _rollout_i in tqdm(range(args.num_rollouts)):
        obs = eval_runner.env_reset()
        for _step_i in tqdm(range(args.num_steps)):
            obs_extracted = eval_runner.extract_from_obs(obs)
            proprioception = obs_extracted.get("proprioception")
            actions = eval_runner.model.generate_actions(
                input_ids=obs_extracted["input_ids"],
                pixel_values=obs_extracted["pixel_values"],
                actions=obs_extracted["actions"],
                attention_mask=obs_extracted["attention_mask"],
                past_mask=obs_extracted["past_mask"],
                proprioception=proprioception,
            )

            actions = eval_runner.denormalize_actions(actions)

            # For each prediction step, we take action_window action steps.
            for action_i in range(eval_runner.num_past_actions, eval_runner.num_past_actions + args.action_window):
                obs = eval_runner.env_step(actions[action_i])
                obs = eval_runner.get_obs_tensor(obs)

                # Updates the past_actions and past_images buffers.
                # We need these buffers to properly construct the input tensors for the model's next inference step.
                # The past_actions buffer has size num_past_actions.
                # The past_images buffer has size num_past_image_timesteps.
                eval_runner.update_action_buffer(actions[action_i])
                eval_runner.update_image_buffer(eval_runner.get_current_images())

                if video_writer is not None:
                    video_writer.append_data(eval_runner.get_image_for_video())

                if eval_runner.check_success() or eval_runner.check_finished():
                    break

            if eval_runner.check_success() or eval_runner.check_finished():
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
