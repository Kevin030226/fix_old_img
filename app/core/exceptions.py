"""Application exceptions shared across layers (plan section 26)."""


class TaskError(RuntimeError):
    """Base class for task-related errors surfaced to the web layer."""


class PipelineFailedError(TaskError):
    """A pipeline stage failed (subprocess exit code or missing output)."""


class StageNotAvailableError(TaskError):
    """The requested stage is not registered in the planner."""


class InvalidRequestError(TaskError):
    """User input failed validation (missing image, oversized, wrong format)."""


class PasswordChangeRequiredError(TaskError):
    """Plan section 21: the account must replace its initial password first."""
