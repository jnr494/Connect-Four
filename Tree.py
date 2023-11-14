import numba
import numpy as np
import numpy.typing as npt

import Connect4Game


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

    def get_action_probabilities(
        self: "Tree",
        game: Connect4Game.Connect4,
        temperature: float = 1,
    ) -> npt.NDArray[np.float64]:
        start_state_hash = game.get_state_hash()
        current_node = self.get_node(start_state_hash)
        no_visits = np.zeros(game.get_number_of_actions())
        no_visits[current_node["actions"]] = current_node["no_visits_actions"]
        no_visits_temperature = no_visits ** (1 / temperature)
        return no_visits_temperature / sum(no_visits_temperature)


@numba.njit
def find_first_in_array(array: npt.ArrayLike, element: float) -> float:
    first_following_action_idx = np.where(array == element)[0]
    if len(first_following_action_idx) > 0:
        return first_following_action_idx[0]
    else:
        return -1
