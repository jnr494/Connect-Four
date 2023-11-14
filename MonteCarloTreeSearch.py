from typing import Callable, Optional, Tuple

import numba
import numpy as np
import numpy.typing as npt

import Connect4Game
from IPlayer import IPlayer


class Node:
    def __init__(
        self: "Node",
        state_hash: int,
        available_actions: list[int],
        next_row_height: npt.NDArray[np.float64],
        priors: npt.NDArray[np.float64],
        prior_win_prediction: float,
    ):
        self.state_hash = state_hash
        self.available_actions = available_actions
        self.next_row_height = next_row_height
        self.priors = priors
        self.prior_win_prediction = prior_win_prediction


class Tree:
    def __init__(self: "Tree") -> None:
        self.nodes: dict = {}

    def new_node(
        self: "Tree",
        state_hash: int,
        available_actions: list[int],
        next_row_height: npt.NDArray[np.float64],
        priors: npt.NDArray[np.float64],
        prior_win_prediction: float,
    ) -> dict:
        new_node = {
            "no_visits": 0,
            "actions": available_actions,
            "actions_idx": {action: idx for idx, action in enumerate(available_actions)},
            "next_row_height": next_row_height,
            "no_visits_actions": np.zeros(len(available_actions), dtype=np.int64),
            "q_values": np.zeros(len(available_actions)),
            "amaf_q_values": np.zeros(len(available_actions)),
            "amaf_no_visits_actions": np.zeros(len(available_actions), dtype=np.int64),
            "next_state_hash": np.zeros(len(available_actions), dtype=np.int64),
            "priors": priors,
            "prior_win_prediction": prior_win_prediction,
        }

        self.nodes[state_hash] = new_node

        return new_node

    def is_node_in_tree(self: "Tree", state_hash: int) -> bool:
        return state_hash in self.nodes

    def get_node(self: "Tree", state_hash: int) -> dict:
        return self.nodes[state_hash]

    def update_node(
        self: "Tree",
        state_hash: int,
        action: int,
        reward: float,
        following_actions: npt.NDArray[np.float64],
        following_row_heights: npt.NDArray[np.float64],
        use_rave: bool = True,
    ) -> None:
        node = self.nodes[state_hash]

        action_idx = node["actions_idx"][action]

        node["no_visits"] += 1
        node["no_visits_actions"][action_idx] += 1
        node["q_values"][action_idx] += (reward - node["q_values"][action_idx]) / node["no_visits_actions"][action_idx]

        # update amaf
        if use_rave:
            for node_action in node["actions"]:
                if len(following_actions) == 0 or node_action == action:
                    break

                first_following_action_idx = find_first_in_array(np.array(following_actions), node_action)

                if first_following_action_idx >= 0:
                    f_row_height = following_row_heights[first_following_action_idx]
                    action_idx = node["actions_idx"][node_action]
                    if f_row_height == node["next_row_height"][action_idx]:
                        node["amaf_no_visits_actions"][action_idx] += 1
                        node["amaf_q_values"][action_idx] += (reward - node["amaf_q_values"][action_idx]) / node[
                            "amaf_no_visits_actions"
                        ][action_idx]

    def update_next_state_hash(self: "Tree", prev_state_hash: int, prev_action: int, current_state_hash: int) -> None:
        prev_node = self.get_node(prev_state_hash)
        prev_action_idx = prev_node["actions_idx"][prev_action]
        prev_node["next_state_hash"][prev_action_idx] = current_state_hash


@numba.njit
def find_first_in_array(array: npt.ArrayLike, element: float) -> float:
    first_following_action_idx = np.where(array == element)[0]
    if len(first_following_action_idx) > 0:
        return first_following_action_idx[0]
    else:
        return -1


def check_game_over(game: Connect4Game.Connect4) -> Tuple[bool, float]:
    reward: float
    game_won: bool
    terminal_bool: bool

    game_won = game.get_current_player() == game.get_winner()
    if game_won:
        terminal_bool = True
        reward = 1.0
    elif len(game.get_available_actions()) == 0:
        # check if game is draw
        terminal_bool = True
        reward = 0.5
    else:
        terminal_bool = False
        reward = 0.0

    return terminal_bool, reward


def select_node_action_ucb1(
    node: dict,
    confidence_value: float,
    rave_param: Optional[float],
    max_bool: bool = True,
) -> int:
    if rave_param is None:
        rave_bool = False
        rave_param = 0
    else:
        rave_bool = True

    return select_node_action_ucb1_numba(
        node["q_values"],
        node["no_visits"],
        node["no_visits_actions"],
        confidence_value,
        node["amaf_q_values"],
        node["amaf_no_visits_actions"],
        rave_bool,
        rave_param,
        node["actions"],
        max_bool,
        node["priors"],
    )


@numba.njit
def select_node_action_ucb1_numba(  # noqa: PLR0913
    q_values: npt.NDArray[np.float64],
    no_visits: list[int],
    no_visits_actions: npt.NDArray[np.float64],
    confidence_value: float,
    amaf_q_values: list[int],
    amaf_no_visits_actions: npt.NDArray[np.float64],
    rave_bool: bool,
    rave_param: float,
    actions: list[int],
    max_bool: bool,
    priors: npt.NDArray[np.float64],
) -> int:
    exploration_term = priors * np.sqrt(no_visits) / (no_visits_actions + 1)

    if rave_bool:
        beta = amaf_no_visits_actions / (
            no_visits_actions + amaf_no_visits_actions + 4 * no_visits_actions * amaf_no_visits_actions * rave_param**2
        )
        adjusted_q_values = (1 - beta) * q_values + beta * amaf_q_values
    else:
        adjusted_q_values = q_values

    if max_bool:
        ucb1_vals = adjusted_q_values + confidence_value * exploration_term
        select_action_idx = np.argmax(ucb1_vals)
    else:
        ucb1_vals = adjusted_q_values - confidence_value * exploration_term
        select_action_idx = np.argmin(ucb1_vals)

    select_action = actions[select_action_idx]

    return select_action


def get_action_probabilities(
    game: Connect4Game.Connect4,
    tree: Tree,
    temperature: float = 1,
) -> npt.NDArray[np.float64]:
    start_state_hash = game.get_state_hash()
    current_node = tree.get_node(start_state_hash)
    no_visits = np.zeros(game.get_number_of_actions())
    no_visits[current_node["actions"]] = current_node["no_visits_actions"]
    no_visits_temperature = no_visits ** (1 / temperature)
    return no_visits_temperature / sum(no_visits_temperature)


def MonteCarloTreeSearch(
    game: Connect4Game.Connect4,
    player: int,
    max_count: int,
    max_depth: int,
    confidence_value: float,
    rave_param: Optional[float],
    rollout_player: IPlayer,
    tree: Optional[Tree] = None,
    evaluator: Optional[Callable] = None,
    rollout_weight: float = 1,
):
    if tree is None:
        tree = Tree()

    if evaluator is None:
        number_of_actions = game.get_number_of_actions()
        evaluator = lambda game: (np.zeros(number_of_actions) + 1 / number_of_actions, 0.5)  # noqa: E731

    use_rave = rave_param is not None

    for _ in range(max_count):
        mcts_single_simulation(
            game,
            tree,
            confidence_value,
            rave_param,
            use_rave,
            max_depth,
            player,
            evaluator,
            rollout_weight,
            rollout_player,
        )

    # find best action
    start_node = tree.get_node(game.get_state_hash())
    best_root_action = select_node_action_ucb1(start_node, 0, None)
    winning_probability = start_node["q_values"][start_node["actions_idx"][best_root_action]]
    return best_root_action, tree, winning_probability


def mcts_single_simulation(
        game: Connect4Game.Connect4,
        tree: Tree,
        confidence_value: float,
        rave_param: float | None,
        use_rave: bool,
        max_depth: int,
        player: int,
        evaluator: Callable,
        rollout_weight: float,
        rollout_player: IPlayer,
) -> None:
    # Reset game and variables for new round
    game_copy = game.copy()
    visited_state_hashes: list[int] = []
    actions: list[int] = []
    new_row_heights: list[int] = []
    terminal_bool = False

    ##selection
    terminal_bool, last_player_reward, current_node = mcts_selection(
        game_copy,
        tree,
        confidence_value,
        rave_param,
        max_depth,
        player,
        evaluator,
        actions,
        new_row_heights,
        visited_state_hashes,
    )

    ##simulation
    if not terminal_bool:
        last_player_reward = mcts_simulation(
            game_copy,
            rollout_weight,
            rollout_player,
            actions,
            new_row_heights,
            current_node,
        )

    player_reward = last_player_reward if (game_copy.get_last_player() == player) else 1 - last_player_reward

    ##backpropagation
    mcts_backpropagation(
        tree,
        player_reward,
        visited_state_hashes,
        actions,
        new_row_heights,
        use_rave,
    )

def mcts_selection(
        game: Connect4Game.Connect4,
        tree: Tree,
        confidence_value: float,
        rave_param: float | None,
        max_depth: int,
        player: int,
        evaluator: Callable,
        actions: list[int],
        new_row_heights: list[int],
        visited_state_hashes: list[int],
    ) -> Tuple[bool, float, dict]:

    terminal_bool, last_player_reward = check_game_over(game)

    while (not terminal_bool) and len(visited_state_hashes) <= max_depth:
        current_state_hash = game.get_state_hash()
        visited_state_hashes.append(current_state_hash)
        no_visited_states = len(visited_state_hashes)

        if no_visited_states > 1:
            tree.update_next_state_hash(visited_state_hashes[-2], actions[-1], current_state_hash)

        ##expansion and stop selection
        if not tree.is_node_in_tree(current_state_hash):
            current_node = mcts_expansion(game, tree, evaluator, current_state_hash)
            break
        else:
            current_node = tree.get_node(current_state_hash)

        ##selection continued
        terminal_bool, last_player_reward = mcts_selection_find_and_perform_action(
            game,
            confidence_value,
            rave_param,
            player,
            current_node,
            actions,
            new_row_heights,
        )

    return terminal_bool, last_player_reward, current_node

def mcts_expansion(
    game: Connect4Game.Connect4,
    tree: Tree,
    evaluator: Callable,
    state_hash: int,
) -> dict:
    clever_available_actions = game.get_clever_available_actions()
    # find priors and win_prediction from evaluator
    priors, win_prediction = evaluator(game)
    filtered_priors = priors[clever_available_actions]
    filtered_priors = filtered_priors / sum(filtered_priors)

    next_row_heights = game.next_row_height[clever_available_actions]
    tree.new_node(
        state_hash,
        clever_available_actions,
        next_row_heights,
        filtered_priors,
        win_prediction,
    )
    current_node = tree.get_node(state_hash)
    return current_node


def mcts_selection_find_and_perform_action(
    game: Connect4Game.Connect4,
    confidence_value: float,
    rave_param: float | None,
    player: int,
    current_node: dict,
    actions: list[int],
    new_row_heights: list[int],
) -> Tuple[bool, float]:
    # get node and find ucb1 optimal action
    selected_action = select_node_action_ucb1(
        current_node,
        confidence_value,
        rave_param,
        max_bool=game.get_current_player() == player,
    )

    actions.append(selected_action)
    new_row_heights.append(game.next_row_height[selected_action])

    # perform action
    game.place_disc(selected_action)
    # check if game is over
    terminal_bool, last_player_reward = check_game_over(game)

    # update turn
    game.next_turn()

    return terminal_bool, last_player_reward


def mcts_simulation(
    game: Connect4Game.Connect4,
    rollout_weight: float,
    rollout_player: IPlayer,
    actions: list[int],
    new_row_heights: list[int],
    current_node: dict,
) -> float:
    if rollout_weight > 0:
        reward = mcts_rollout(
            game,
            rollout_player,
            actions,
            new_row_heights,
        )

    reward = rollout_weight * reward + (1 - rollout_weight) * current_node["prior_win_prediction"]
    return reward


def mcts_rollout(
    game: Connect4Game.Connect4,
    rollout_player: IPlayer,
    actions: list[int],
    new_row_heights: list[int],
) -> float:
    terminal_bool, last_player_reward = check_game_over(game)
    while not terminal_bool:
        # get available actions
        clever_available_actions = game.get_clever_available_actions()

        # get and simulate action
        sim_action = rollout_player.make_action(game, clever_available_actions)
        actions.append(sim_action)
        new_row_heights.append(game.next_row_height[sim_action])
        game.place_disc(sim_action)

        # check if game is over
        terminal_bool, last_player_reward = check_game_over(game)

        # update turn
        game.next_turn()

    return last_player_reward


def mcts_backpropagation(
    tree: Tree,
    reward: float,
    visited_states: list[int],
    actions: list[int],
    new_row_heights: list[int],
    use_rave: bool,
) -> None:
    # update player rewards
    for idx in range(0, len(visited_states)):
        tree.update_node(
            visited_states[idx],
            actions[idx],
            reward,
            following_actions=np.array(actions[idx + 2 :: 2]),
            following_row_heights=np.array(new_row_heights[idx + 2 :: 2]),
            use_rave=use_rave,
        )
