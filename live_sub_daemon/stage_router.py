from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence

from .config import AppConfig, ProviderRuntimeConfig
from .llm_client import StageResult, XAIClient


@dataclass
class RoutedStageResult:
    result: StageResult
    provider: str
    model: str


@dataclass
class RoutedTranslateResult:
    final: RoutedStageResult
    primary: RoutedStageResult
    fallback: Optional[RoutedStageResult]
    fallback_used: bool


class StageRouter:
    def __init__(self, config: AppConfig):
        self._cfg = config
        self._clients: Dict[str, XAIClient] = {}
        for provider, runtime in config.llm.provider_settings.items():
            if not runtime.api_key:
                continue
            self._clients[provider] = XAIClient(api_key=runtime.api_key, base_url=runtime.base_url)

    def route_summary(self) -> Dict[str, str]:
        return {
            "correct_provider": self._cfg.llm.correct_provider,
            "translate_provider": self._cfg.llm.translate_provider,
            "translate_fallback_provider": self._cfg.llm.translate_fallback_provider,
        }

    def run_correct_batch(
        self,
        *,
        stage_name: str,
        system_prompt: str,
        items: Sequence[dict[str, str]],
        max_retries: int,
        retry_backoff_sec: float,
        temperature: float,
        use_fast_model: bool = False,
    ) -> RoutedStageResult:
        runtime = self._provider_config(self._cfg.llm.correct_provider)
        model = runtime.fast_correct_model if use_fast_model and runtime.fast_correct_model else runtime.correct_model
        result = self._client(runtime.provider).run_batch(
            model=model,
            stage_name=stage_name,
            system_prompt=system_prompt,
            items=items,
            timeout_sec=self._cfg.llm.correct_timeout_sec,
            max_retries=max_retries,
            retry_backoff_sec=retry_backoff_sec,
            temperature=temperature,
        )
        return RoutedStageResult(result=result, provider=runtime.provider, model=model)

    def run_translate_batch(
        self,
        *,
        stage_name: str,
        system_prompt: str,
        items: Sequence[dict[str, str]],
        max_retries: int,
        retry_backoff_sec: float,
        temperature: float,
        use_fast_model: bool = False,
        translation_options: Optional[dict[str, object]] = None,
    ) -> RoutedTranslateResult:
        primary_runtime = self._provider_config(self._cfg.llm.translate_provider)
        primary_model = (
            primary_runtime.fast_translate_model
            if use_fast_model and primary_runtime.fast_translate_model
            else primary_runtime.translate_model
        )
        primary = RoutedStageResult(
            result=self._client(primary_runtime.provider).run_batch(
                model=primary_model,
                stage_name=stage_name,
                system_prompt=system_prompt,
                items=items,
                timeout_sec=self._cfg.llm.translate_timeout_sec,
                max_retries=max_retries,
                retry_backoff_sec=retry_backoff_sec,
                temperature=temperature,
                translation_options=translation_options,
            ),
            provider=primary_runtime.provider,
            model=primary_model,
        )

        if primary.result.ok:
            return RoutedTranslateResult(
                final=primary,
                primary=primary,
                fallback=None,
                fallback_used=False,
            )

        if not self._cfg.llm.translate_fallback_on_error:
            return RoutedTranslateResult(
                final=primary,
                primary=primary,
                fallback=None,
                fallback_used=False,
            )

        fallback_runtime = self._provider_config(self._cfg.llm.translate_fallback_provider)
        fallback_model = (
            fallback_runtime.fast_translate_model
            if use_fast_model and fallback_runtime.fast_translate_model
            else fallback_runtime.translate_model
        )
        fallback = RoutedStageResult(
            result=self._client(fallback_runtime.provider).run_batch(
                model=fallback_model,
                stage_name=f"{stage_name}_fallback",
                system_prompt=system_prompt,
                items=items,
                timeout_sec=self._cfg.llm.translate_fallback_timeout_sec,
                max_retries=0,
                retry_backoff_sec=retry_backoff_sec,
                temperature=temperature,
                translation_options=translation_options,
            ),
            provider=fallback_runtime.provider,
            model=fallback_model,
        )
        final = fallback if fallback.result.ok else primary
        return RoutedTranslateResult(
            final=final,
            primary=primary,
            fallback=fallback,
            fallback_used=True,
        )

    def _provider_config(self, provider: str) -> ProviderRuntimeConfig:
        runtime = self._cfg.llm.provider_settings.get(provider)
        if runtime is None:
            raise ValueError(f"Missing provider config: {provider}")
        return runtime

    def _client(self, provider: str) -> XAIClient:
        client = self._clients.get(provider)
        if client is None:
            runtime = self._provider_config(provider)
            raise ValueError(f"Provider '{provider}' is missing API key ({runtime.api_key_env})")
        return client
