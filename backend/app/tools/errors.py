class ToolError(Exception):
    """Base class for internal tool errors."""


class ToolAuthorizationError(ToolError):
    """Raised when a tool call is not authorized."""


class ToolInputValidationError(ToolError):
    """Raised when tool input fails schema validation."""


class ToolUnavailableError(ToolError):
    """Raised when a registered tool is not executable yet."""


class UnknownToolError(ToolError):
    """Raised when a tool is not registered."""
