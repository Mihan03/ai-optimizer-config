"""KHL match model v2.0: регуляризованный Пуассон + поправки на ничьи и пустые ворота,
смешивание с рынком, отбор ставок по EV и доле Келли."""
from .config import Params
from .scores import MatchModel

__all__ = ["Params", "MatchModel"]
