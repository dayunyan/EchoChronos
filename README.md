# EchoChronos 🥰

### Description
EchoChronos (Time Echo Chronos) is a multi-modal style conversational AI assistant based on a large language model, designed to provide users with a new style of conversational experience. The AI integrates RAG, TTS, and other technologies to enable real-time interaction with users, allowing them to immerse themselves in the charm of classic dialogues and dialogues with history. 🥸

![lindaiyu](./assert/image.png)

### Software Architecture
```
EchoChronos
├─ ChatStyle  # Conversation style module
├─ managers   # Used to provide interfaces for various modules
│  ├─ __init__.py
│  ├─ connect.py  # Connection manager, currently only supports WebSocket
│  ├─ constants.py  # Constants
│  ├─ model.py  # Style dialogue model manager
│  ├─ rag.py  # RAG model manager
│  ├─ runner.py  # Runner manager, used for writing inference logic
│  └─ tts.py  # TTS model manager
├─ RAG  # RAG module
├─ TTS  # TTS module
├─ utils  # Toolkit
├─ inference_torch.py  # Inference code using PyTorch
├─ inference.py  # Inference code using MindSpore
├─ launch.py  # Project entry point
├─ README.en.md
└─ README.md
```

### Installation

python>=3.10

#### Install ffmpeg

``` shell
conda install ffmpeg
```

#### Install MindNLP

``` shell
pip install git+https://github.com/mindspore-lab/mindnlp.git
```

#### Install other dependencies

``` shell
git clone --recursive https://gitee.com/xujunda2024/echochronos.git
cd echochronos
pip install -r requirements.txt
```

### Instructions

1. Set up a Python environment ☝️🤓 

2. After successfully installing the dependencies, prepare the configuration file in the format of `examples/infer_qwen2_lora_fp32.yaml` (be sure to modify the parameters in the configuration file according to your needs). 

3. Start the GPT-SOVITS service.

    - Prepare the model: For details, please refer to the [README.md](./TTS/GPT-SoVITS-main/README.md) file of the GPT-SOVITS project.

    - Start the service:
        ``` shell
        cd TTS/GPT-SoVITS-main/GPT_SOVITS
        python Server.py
        ```

4. Currently, this project provides three modes of operation, which can be changed by modifying the "isTerminal", "isWebsocket", and "isWebUI" parameters in the YAML file (Replace `<your_yaml_path>` with your YAML-formatted configuration file).
 
    - Terminal: `python launch.py <your_yaml_path>`

    - WebSocket: `python launch.py <your_yaml_path>`

    - WebUI (recommended 🤩): `streamlit run launch.py <your_yaml_path>`


### MindSpore Users 🤯

#### GPU Users 

- It is recommended to use CUDA 11.6 and cuDNN. 

- If you encounter `[ERROR] libcuda.so (needed by mindspore-gpu) is not found.`, please `export CUDA_HOME=/path/to/your/cuda` 

#### Video Memory 

- Due to the accuracy issues of the Qwen2 models in the MindNLP, inference can only be performed using float32, with memory consumption around 46G.

#### Instructions for MindSpore Users

- MindSpore has two inference script entry points, and the startup methods are as follows:

    - `python inference_ms.py --isTerminal` or `python inference_ms.py --isWebsocket`

    - `streamlit run webui.py` (recommended 🤩)

