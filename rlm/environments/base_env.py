from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable

from rlm.core.types import REPLResult


class BaseEnv(ABC):
    """
    Base REPL-like environment that the RLM uses to interact with. The primary types are isolated and non-isolated,
    where isolated environments are on a separate machine from the LM.
    """

    def __init__(self, persistent: bool = False, **kwargs):
        self.persistent = persistent
        self.kwargs = kwargs

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class IsolatedEnv(BaseEnv, ABC):
    """
    These environments (e.g. Prime Envs, Modal Envs) sit on a completely separate machine from the LM,
    guaranteeing complete isolation from the LM process.
    """

    def __init__(self, persistent: bool = False, **kwargs):
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


class NonIsolatedEnv(BaseEnv, ABC):
    """
    These environments run on the same machine as the LM, and provide different levels of isolation
    depending on the choice of environment. The simplest, default is a local Python REPL that runs
    as a subprocess.
    """

    def __init__(self, persistent: bool = False, **kwargs):
        super().__init__(persistent=persistent, **kwargs)

    @abstractmethod
    def setup(self):
        raise NotImplementedError

    @abstractmethod
    def load_context(self, context_payload: dict | list | str):
        raise NotImplementedError

    @abstractmethod
    def execute_code(self, code: str) -> REPLResult:
        raise NotImplementedError


@runtime_checkable
class SupportsPersistence(Protocol):
    """Protocol for environments that support persistent multi-turn sessions.

    Use isinstance(env, SupportsPersistence) to check if an environment
    supports persistence capabilities.
    """

    def update_handler_address(self, address: tuple[str, int]) -> None: ...
    def add_context(
        self, context_payload: dict | list | str, context_index: int | None = None
    ) -> int: ...
    def get_context_count(self) -> int: ...
    def add_history(
        self, message_history: list[dict[str, Any]], history_index: int | None = None
    ) -> int: ...
    def get_history_count(self) -> int: ...
