class CreditRiskError(Exception):
    def __init__(self, message: str, status_code: int = 500) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class DataIngestionError(CreditRiskError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=500)


class DataValidationError(CreditRiskError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=422)


class ModelNotLoadedError(CreditRiskError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=503)


class InferenceExecutionError(CreditRiskError):
    def __init__(self, message: str) -> None:
        super().__init__(message=message, status_code=500)
