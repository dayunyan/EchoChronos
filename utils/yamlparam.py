import yaml


class YAMLParamHandler:
    def __init__(self, yaml_file_path):
        """
        初始化函数，接收YAML文件路径参数
        """
        self.yaml_file_path = yaml_file_path
        self.yaml_params = self.load_yaml_params()

    def load_yaml_params(self):
        """
        根据传入的YAML文件路径加载并返回参数内容（以字典形式）
        如果文件不存在或者解析YAML出错，进行相应的异常处理
        """
        try:
            with open(self.yaml_file_path, "r") as file:
                self.yaml_params = yaml.safe_load(file)
        except FileNotFoundError:
            print(f"Error: File {self.yaml_file_path} not found.")
            raise
        except yaml.YAMLError as e:
            print(f"Error parsing YAML file: {e}")
            raise

        return self.yaml_params

    def check_yaml_params(self, required_keys):
        """
        检查YAML参数中是否包含所有必需的键
        """
        missing_keys = [key for key in required_keys if key not in self.yaml_params]
        if missing_keys:
            print(f"Missing required keys in YAML file: {missing_keys}")
            raise KeyError(f"Missing required keys: {missing_keys}")

    def get_yaml_params(self):
        """
        返回加载的YAML参数内容
        """

        return self.yaml_params
