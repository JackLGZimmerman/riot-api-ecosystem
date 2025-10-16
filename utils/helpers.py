from contextlib import contextmanager
from typing import Type, TypeVar, Generator

E = TypeVar("E", bound=BaseException)

@contextmanager
def reraise(exc_type: Type[E], msg: str) -> Generator[None, None, None]:
    try:
        yield
    except Exception as e:
        raise exc_type(msg) from e