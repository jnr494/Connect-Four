from abc import ABC, abstractmethod

from Connect4Game import Connect4

class IPlayer(ABC):
    
    @abstractmethod
    def make_action(self, game: Connect4, available_actions: list[int]) -> int:
        pass
    
    @abstractmethod
    def reset(self):
        pass