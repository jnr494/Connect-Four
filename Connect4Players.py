import logging
from typing import Tuple

import numba  # type : ignore
import numpy as np
import numpy.typing as npt

from MonteCarloTreeSearch import MonteCarloTreeSearchEngine
from Tree import Tree
from ConfigHandler import MCTSPlayerConfig
from Connect4Game import Connect4
from IPlayer import IPlayer
from LoggerHandler import LoggerHandler


class RandomPlayer(IPlayer):
    _name: str

    def __init__(self: "RandomPlayer") -> None:
        self._name = "random"

    def make_action(self: "RandomPlayer", game: Connect4, available_actions: list[int]) -> int:
        return make_random_choice(available_actions)

    def reset(self: "RandomPlayer") -> None:
        pass

    def get_name(self: "RandomPlayer") -> str:
        return self._name


class DQPlayer(IPlayer):
    def __init__(self: "DQPlayer", qmodel, epsilon: float) -> None:
        self.qmodel = qmodel
        self.epsilon = epsilon
        self.RandomPlayer = RandomPlayer()

    def make_action(self: "DQPlayer", game: Connect4, available_actions: list[int]) -> int:
        uniform: float = np.random.Generator.random()
        if uniform < self.epsilon:
            return self.RandomPlayer.make_action(game, available_actions)
        else:
            q_values = self.QModel.predict(np.expand_dims(game._get_board().flatten(), axis=0))
            q_values_available = q_values[0, available_actions]
            action = available_actions[np.argmax(q_values_available)]
            return action

    def reset(self: "DQPlayer") -> None:
        pass


class MCTSPlayer(IPlayer):
    _name: str
    _game: Connect4
    _player: int
    _next_player: int
    _mcts_config: MCTSPlayerConfig
    winning_probability: float | None
    _rollout_player: RandomPlayer = RandomPlayer()
    _tree: Tree
    _logger: logging.Logger
    _mcts_engine: MonteCarloTreeSearchEngine

    def __init__(
        self: "MCTSPlayer",
        game: Connect4,
        player: int,
        next_player: int,
        mcts_config: MCTSPlayerConfig,
        logger_handler: LoggerHandler,
    ) -> None:
        self._game = game
        self._player = player
        self._next_player = next_player
        self._mcts_config = mcts_config
        self._logger = logger_handler.get_logger(type(self).__name__)

        self._name = self._mcts_config.name

        self.reset()

    def make_action(self: "MCTSPlayer", game: Connect4, available_actions: list[int]) -> int:
        if not self._mcts_config.reuse_tree:
            self._get_new_tree()

        self._game = game

        #Perform monte carlo tree search
        self._mcts_engine.perform_search(self._mcts_config.max_count)

        if not self._mcts_config.randomize_action:
            #Get best action and winning probability
            best_action, winning_probability = self._mcts_engine.get_best_root_action()

            self.winning_probability = winning_probability
            if self.winning_probability is not None:
                self._logger.debug(
                    f"Name=[{self._mcts_config.name}] found best action=[{best_action}] has estimated probability of winning [{round(self.winning_probability*100,2)}%]",  # noqa: E501
                )

            return best_action

        else:
            action_probabilities = self._tree.get_action_probabilities(self._game, temperature=1)
            action = np.random.choice(self._game.get_number_of_actions(), p=action_probabilities)  # noqa: NPY002
            return action

    def reset(self: "MCTSPlayer") -> None:
        self._evaluator = standard_evaluator
        self.winning_probability = None
        self._get_new_tree()
        self._get_new_mcts_engine()

    def get_name(self: "MCTSPlayer") -> str:
        return self._name

    def _get_new_tree(self: "MCTSPlayer") -> None:
        self._tree = Tree()
        if hasattr(self, "_mcts_engine"):
            self._mcts_engine.set_tree(self._tree)

    def _get_new_mcts_engine(self: "MCTSPlayer") -> None:
        self._mcts_engine = MonteCarloTreeSearchEngine(
            self._game,
            self._player,
            self._mcts_config,
            self._rollout_player,
            self._tree,
            self._evaluator,
        )


@numba.njit
def make_random_choice(available_actions: list[int]) -> int:
    return np.random.choice(available_actions)  # noqa: NPY002


def standard_evaluator(game: Connect4) -> Tuple[npt.NDArray[np.float64], float]:
    number_of_actions = game.get_number_of_actions()
    return standard_evaluator_numba(number_of_actions)


@numba.njit
def standard_evaluator_numba(number_of_actions: int) -> Tuple[npt.NDArray[np.float64], float]:
    return (np.ones(number_of_actions) / number_of_actions, 0.5)
