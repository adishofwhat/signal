"""
Unified LLM client — swap provider without touching any other code.
Supports Claude (Anthropic), OpenAI, and Gemini.
"""
from enum import Enum


class LLMProvider(Enum):
    CLAUDE = "claude"
    OPENAI = "openai"
    GEMINI = "gemini"


_DEFAULT_MODELS = {
    LLMProvider.CLAUDE: "claude-sonnet-4-6",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.GEMINI: "gemini-2.5-flash",
}


class LLMClient:
    """
    Unified interface for LLM calls across providers.
    All methods return raw text; JSON parsing is the caller's responsibility.
    """

    def __init__(
        self,
        provider: LLMProvider | str = LLMProvider.CLAUDE,
        api_key: str = "",
        model: str | None = None,
    ):
        if isinstance(provider, str):
            provider = LLMProvider(provider.lower())
        self.provider = provider
        self.api_key = api_key
        self.model = model or _DEFAULT_MODELS[provider]

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = 4096,
        json_mode: bool = False,
    ) -> str:
        """
        Generate a response from the configured LLM provider.
        json_mode=True hints to the provider to return valid JSON (OpenAI only;
        for other providers it's enforced via prompt wording).
        """
        if self.provider == LLMProvider.CLAUDE:
            return self._call_claude(system_prompt, user_prompt, max_tokens)
        elif self.provider == LLMProvider.OPENAI:
            return self._call_openai(system_prompt, user_prompt, max_tokens, json_mode)
        elif self.provider == LLMProvider.GEMINI:
            return self._call_gemini(system_prompt, user_prompt, max_tokens)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    def _call_claude(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
        import anthropic
        client = anthropic.Anthropic(api_key=self.api_key)
        response = client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        return response.content[0].text

    def _call_openai(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int,
        json_mode: bool,
    ) -> str:
        from openai import OpenAI
        client = OpenAI(api_key=self.api_key)
        kwargs: dict = {"max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            **kwargs,
        )
        return response.choices[0].message.content

    # def _call_gemini(self, system_prompt: str, user_prompt: str, max_tokens: int) -> str:
    #     import google.generativeai as genai
    #     genai.configure(api_key=self.api_key)
    #     model = genai.GenerativeModel(
    #         model_name=self.model,
    #         system_instruction=system_prompt,
    #     )
    #     response = model.generate_content(
    #         user_prompt,
    #         generation_config=genai.types.GenerationConfig(
    #             max_output_tokens=4096,
    #             response_mime_type="application/json",  # forces valid JSON output
    #         )
    #     )
    #     return response.text

    def _call_gemini(self, system_prompt, user_prompt, max_tokens=4096):
        import google.generativeai as genai
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(
            model_name=self.model,
            system_instruction=system_prompt,
            generation_config=genai.types.GenerationConfig(
                max_output_tokens=8192,
                response_mime_type="application/json",
            ),
        )
        response = model.generate_content(user_prompt)
        return response.text
