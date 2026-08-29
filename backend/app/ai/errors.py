class AIProviderError(Exception):
    """Base class for internal AI provider errors."""


class AIConfigurationError(AIProviderError):
    """Raised when the AI provider is not configured correctly."""


class AIProviderTimeoutError(AIProviderError):
    """Raised when an AI provider request exceeds the configured timeout."""


class AIProviderTransientError(AIProviderError):
    """Raised when a transient provider failure remains after retries."""


class AIProviderResponseError(AIProviderError):
    """Raised when the provider returns an unusable response."""


class StructuredOutputValidationError(AIProviderError):
    """Raised when structured model output fails Pydantic validation."""
