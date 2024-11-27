from utils.argparser import get_args, check_args
from managers.runner import RunnerManager

if __name__ == "__main__":
    args = get_args()
    check_args(args)
    runner = RunnerManager(**vars(args))
    runner.run()
