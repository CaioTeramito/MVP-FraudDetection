class ArtifactNotFoundError(FileNotFoundError):
    """Raised when the serialized model artifact cannot be found."""


class DatasetValidationError(ValueError):
    """Raised when a dataset cannot be validated for training."""
