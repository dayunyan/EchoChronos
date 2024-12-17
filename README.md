# EchoChronos 🥰

#### Description
EchoChronos (Time Echo Chronos) is a multi-modal style conversational AI assistant based on a large language model, designed to provide users with a new style of conversational experience. The AI integrates RAG, TTS, and other technologies to enable real-time interaction with users, allowing them to immerse themselves in the charm of classic dialogues and dialogues with history. 🥸

![lindaiyu](./assert/image.png)

#### Software Architecture
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

#### Installation

python==3.11
``` shell
git clone --recursive https://gitee.com/xujunda2024/echochronos.git
cd echochronos
conda install ffmpeg
pip install -r requirements.txt
```

#### Instructions

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
    - WebUI (Recommendation): `streamlit run launch.py <your_yaml_path>`

> [!TIP]
> MindSpore currently only supports two execution modes: Terminal and WebSocket, with the entry point being inference.py.
