from openai import OpenAI
import json

class GPTTextProcessor:
    def __init__(self, config_path="../config.json"):
        with open(config_path, "r") as f:
            config = json.load(f)
        self.client = OpenAI(api_key=config["openai_api_key"])

    def process_text(self, text):
        prompt = f"""
        This is a transcribed text from spoken English, which may contain some minor errors. Please first correct any grammatical or expression mistakes in the content. Then, **extract all the important action units** and output them in a list format.
        Original transcribed text:
        {text}
        Output format:
        Corrected text: xxx
        List of action units: [Action 1, Action 2, ...]
        """

        response = self.client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an assistant who is good at English text processing."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2, #output conservative answer
            max_tokens=500
        )
        return response.choices[0].message.content