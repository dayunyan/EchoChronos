import sys

# from utils.argparser import get_args, check_args
from utils.yamlparam import YAMLParamHandler
from managers.runner import RunnerManager


if __name__ == "__main__":
    # args = get_args()
    # check_args(args)
    if len(sys.argv) < 2:
        print("Please provide the YAML file path.")
        sys.exit(1)
    yaml_processor = YAMLParamHandler(sys.argv[1])
    yaml_params = yaml_processor.load_yaml_params()
    # 在这里可以根据从YAML文件中读取到的参数进行后续的项目相关操作
    # print(yaml_params)
    runner = RunnerManager(
        runner_config=yaml_params.get("runner"),
        model_config=yaml_params.get("model"),
        rag_config=yaml_params.get("rag_config"),
    )
    runner.run()
