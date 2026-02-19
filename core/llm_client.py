import google.generativeai as genai

class LLMClient:
    def __init__(self, api_key: str, model_name: str):
        genai.configure(api_key=api_key)
        self.model_name = model_name
        self.model = genai.GenerativeModel(model_name)

    def generate(self, system_prompt: str, context: str, user_prompt: str):
        prompt = f"{system_prompt}\n\n{context}\n\nUSER:\n{user_prompt}"

        response = self.model.generate_content(
            prompt,
            generation_config={
                "temperature": 0.6,
                "max_output_tokens": 3000,
            },
        )
        return response.text