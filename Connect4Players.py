
from typing import Tuple

import numba
import numpy as np

import MonteCarloTreeSearch
from Connect4Game import Connect4
from IPlayer import IPlayer


class RandomPlayer(IPlayer):
    def __init__(self: "RandomPlayer") -> None:
        pass

    def make_action(self: "RandomPlayer", game: Connect4, available_actions: list[int]) -> None:
        return make_random_choice(available_actions)

    def reset(self: "RandomPlayer") -> None:
        pass


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
            q_values = self.QModel.predict(np.expand_dims(game.Board.flatten(), axis=0))
            q_values_available = q_values[0, available_actions]
            action = available_actions[np.argmax(q_values_available)]
            return action

    def reset(self: "DQPlayer") -> None:
        pass


class MCTSPlayer(IPlayer):
    def __init__(
        self: "MCTSPlayer",
        game: Connect4,
        player: int,
        next_player: int,
        max_count: int,
        max_depth: int,
        confidence_value: float,
        rave_param: float | None = None,
        reuse_tree: bool = True,
        randomize_action: bool = False,
    ) -> None:
        self.game = game
        self.player = player
        self.next_player = next_player
        self.max_count = max_count
        self.max_depth = max_depth
        self.confidence_value = confidence_value
        self.rave_param = rave_param
        self.RandomPlayer = RandomPlayer()
        self.reuse_tree = reuse_tree
        self.randomize_action = randomize_action

        self.reset()

    def make_action(self: "MCTSPlayer", game: Connect4, available_actions: list[int]) -> int:
        self.game = game
        # important that env_state comes from game.
        best_action, tree, winning_probability = MonteCarloTreeSearch.MonteCarloTreeSearch(
            self.game,
            self.player,
            self.next_player,
            self.max_count,
            self.max_depth,
            self.confidence_value,
            self.rave_param,
            self.RandomPlayer,
            self.tree,
        )

        self.winning_probability = winning_probability

        if self.reuse_tree:
            self.tree = tree

        if self.randomize_action:
            action_probabilities = MonteCarloTreeSearch.get_action_probabilities(self.game, tree, temperature=1)
            action: int = np.random.Generator.choice(self.game.no_cols, p=action_probabilities)
            return (action, action_probabilities)
        else:
            return best_action

    def reset(self: "MCTSPlayer") -> None:
        self.tree = None
        self.winning_probability = None

    def get_optimal_actions_qvalues(self: "MCTSPlayer") -> Tuple[list, list, list, list]:
        best_actions, q_values, amaf_q_values, nodes = MonteCarloTreeSearch.get_optimal_tree_actions(
            self.game,
            self.tree,
            self.player,
            self.next_player,
        )
        return best_actions, q_values, amaf_q_values, nodes


@numba.njit
def make_random_choice(available_actions: list[int]) -> int:
    return np.random.choice(available_actions)  # noqa: NPY002
