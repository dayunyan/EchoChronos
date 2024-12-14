import requests


class TTSManager:
    def __init__(
        self,
        server_url: str = "http://172.24.244.174:5000/tts",
    ):
        self.server_url = server_url

    def get_data(
        self,
        character,
        target_text,
        prompt_language="中文",
        text_language="中文",
        how_to_cut="不切",
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
    ):

        return {
            "character": character,
            # 'prompt_text': prompt_text,
            "prompt_language": prompt_language,  # 参考文本的语言
            "text": target_text,
            "text_language": text_language,  # 目标文本的语言
            "how_to_cut": how_to_cut,  # 文本切分方式
            "top_k": top_k,  # Top-K 参数
            "top_p": top_p,  # Top-P 参数
            "temperature": temperature,  # 温度参数
            "ref_free": ref_free,  # 是否使用参考音频
        }

    def process(
        self,
        character,
        target_text,
        prompt_language="中文",
        text_language="中文",
        how_to_cut="不切",
        top_k=20,
        top_p=0.6,
        temperature=0.6,
        ref_free=False,
    ):
        data = self.get_data(
            character,
            target_text,
            prompt_language,
            text_language,
            how_to_cut,
            top_k,
            top_p,
            temperature,
            ref_free,
        )
        response = requests.post(self.server_url, data=data)
        if response.status_code == 200:
            audio_content = response.content
        else:
            print(f"Error: {response.status_code}, {response.text}")

        return audio_content
