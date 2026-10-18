import argparse
from agcd.AGCD import run
import os
import json
from utils.MyUtils import get_par
root_dir = os.path.dirname(os.path.abspath(__file__))

if __name__ == "__main__":
    # Parser
    parser = argparse.ArgumentParser(description='PyTorch args')
    parser.add_argument('--env', type=str, default='Walker-v0',
                        help='env')
    parser.add_argument('--seed', type=int, default=101,
                        help='random seed')
    parser.add_argument('--target_size', type=int, default=5,
                        help='target design space, default 5x5')
    parser.add_argument('--threads_num', type=int, default=20,
                        help='number of CPU threads')
    parser.add_argument('--max_iters', type=int, default=50000,
                        help='')
    parser.add_argument('--pop_size', type=int, default=20,
                        help='')
    parser.add_argument('--train_iters', type=int, default=40,
                        help='')
    parser.add_argument('--total_step', type=int, default=1000,
                        help='')
    parser.add_argument('--total_maturity', type=int, default=25,
                        help='')
    parser.add_argument('--save_to', type=str, default='',
                        help='save_to')
    parser.add_argument('--use_attention', action='store_true',
                        help='use attention')
    parser.add_argument('--inherit', action='store_true',
                        help='')
    parser.add_argument('--use_mfp', action='store_true',
                        help='')

    args = parser.parse_args()
    args.seed = 101
    args.target_size = 5
    args.threads_num = 20
    args.pop_size = 20
    args.use_attention = True
    args.total_maturity = 25
    args.inherit = True
    args.use_mfp = True
    env_list= ["Walker-v0","DownStepper-v0"]
    num=1
    for env in env_list:
        max_eva, tc = get_par(env)
        args.env=env
        args.train_iters = int(tc//args.total_maturity)
        args.total_step=tc
        args.max_iters=max_eva*tc
        temp_save_to=os.path.join(root_dir, "result","AGCD", args.env)
        if not args.use_attention:
            temp_save_to += "_noA"
        if not args.use_mfp:
            temp_save_to += "_noM"
        if not args.inherit:
            temp_save_to += "_noI"
        for i in range(num):
            args.save_to=os.path.join(temp_save_to, str(i))
            if not os.path.exists(args.save_to):
                os.makedirs(args.save_to)
                args_file_path = os.path.join(args.save_to, "config.json")
                with open(args_file_path, "w") as f:
                    json.dump(vars(args), f, indent=4)
            else:
                continue
            run(args)

