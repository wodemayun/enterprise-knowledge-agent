class DomainError(Exception):
    """Base class for errors that are safe to translate into API responses."""


class UnsupportedDocumentError(DomainError):
    """Raised when an uploaded document type is not supported."""


class EmptyDocumentError(DomainError):
    """Raised when no searchable text can be extracted."""


class ConfigurationError(DomainError):
    """Raised when required runtime configuration is missing or invalid."""


class DependencyUnavailableError(DomainError):
    """Raised when a required database or external service is unavailable."""

