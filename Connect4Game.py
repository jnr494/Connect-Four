import copy
from typing import Optional

import matplotlib.pyplot as plt
import numba  # type: ignore
import numpy as np
import numpy.typing as npt

from GameTurnHandler import GameTurnHandler


class Connect4:
    _game_turn_handler: GameTurnHandler
    _winner: int | None
    _board: npt.NDArray[np.float64]
    next_row_height: npt.NDArray[np.float64]
    _last_player: int | None
    _round: int

    def __init__(
        self: "Connect4",
        game: Optional["Connect4"] = None,
        game_turn_handler: Optional[GameTurnHandler] = None,
    ) -> None:
        self._no_rows = 6
        self._no_cols = 7
        self.fig: plt.Figure | None = None

        if game_turn_handler is not None:
            self._game_turn_handler = game_turn_handler

        self.reset(game=game)

    def get_number_of_actions(self: "Connect4") -> int:
        return self._no_cols

    def get_max_rounds(self: "Connect4") -> int:
        return self._no_cols * self._no_rows

    def place_disc(self: "Connect4", col: int) -> bool:
        return self._place_disc(col, self._game_turn_handler.get_current_player_value())

    def _place_disc(self: "Connect4", col: int, player: int) -> bool:
        # place disc
        row = self.next_row_height[col]
        self._board[row, col] = player

        # update next_row_height
        self.next_row_height[col] += 1
        #update last_player
        self._last_player = player
        #update round
        self._round += 1

        # check for win
        if self._winner is not None:
            return True
        elif self._current_winning_possibilities[player][row, col] == 1:
            self._winner = player
            return True
        else:
            # update current winning_possibilities
            update_winning_possibilities(
                self._get_board(),
                self._current_winning_possibilities[player],
                player,
                col,
                row,
            )
            return False

    def get_last_player(self: "Connect4") -> int | None:
        return self._last_player

    def get_winner(self: "Connect4") -> int | None:
        return self._winner

    def get_round(self: "Connect4") -> int:
        return self._round

    def is_draw(self: "Connect4") -> int:
        return self.get_round() == self.get_max_rounds()

    def _get_board(self: "Connect4") -> npt.NDArray[np.float64]:
        return self._board

    def get_state_hash(self: "Connect4") -> int:
        return hash(self._get_board().tobytes())

    def next_turn(self: "Connect4") -> None:
        self._game_turn_handler.next_turn()

    def get_current_player(self: "Connect4") -> int:
        return self._game_turn_handler.get_current_player_value()

    def get_current_player_turn(self: "Connect4") -> int:
        return self._game_turn_handler.get_current_player_turn()

    def reset(self: "Connect4", game: Optional["Connect4"] = None) -> None:
        if game is None:
            self._board = np.zeros((self._no_rows, self._no_cols))
            self.next_row_height = np.zeros((self._no_cols,), dtype=int)
            self._current_winning_possibilities = {
                1: np.zeros((self._no_rows, self._no_cols)),
                -1: np.zeros((self._no_rows, self._no_cols)),
            }
            self._winner = None
            self._last_player = None
            self._round = 0
            if hasattr(self, "_game_turn_handler"):
                self._game_turn_handler.reset()
        else:
            self._board = copy.deepcopy(game._get_board())
            self.next_row_height = copy.deepcopy(game.next_row_height)
            self._current_winning_possibilities = copy.deepcopy(game._current_winning_possibilities)
            self._winner = game._winner
            self._last_player = game._last_player
            self._round = game.get_round()
            self._game_turn_handler = game.get_turn_handler().copy()

    def get_turn_handler(self: "Connect4") -> GameTurnHandler:
        return self._game_turn_handler

    def copy(self: "Connect4") -> "Connect4":
        return Connect4(self)

    def get_available_actions(self: "Connect4") -> list[int]:
        return get_available_actions_numba(self._get_board())

    def get_clever_available_actions(self: "Connect4") -> list[int]:
        return self._get_clever_available_actions(
            self._game_turn_handler.get_current_player_value(),
            self._game_turn_handler.get_next_player_value(),
        )

    def _get_clever_available_actions(self: "Connect4", player: int, next_player: int) -> list[int]:
        winning_actions = find_available_winning_actions(
            self._current_winning_possibilities[player],
            self.next_row_height,
            self._no_cols,
            self._no_rows,
        )
        if len(winning_actions) > 0:
            return winning_actions

        must_block_actions = find_available_winning_actions(
            self._current_winning_possibilities[next_player],
            self.next_row_height,
            self._no_cols,
            self._no_rows,
        )
        if len(must_block_actions) > 0:
            return must_block_actions

        all_available_actions = self.get_available_actions()
        filtered_actions = exclude_must_avoid_actions(
            all_available_actions,
            self._current_winning_possibilities[next_player],
            self.next_row_height,
            self._no_rows,
        )
        if len(filtered_actions) > 0:
            return filtered_actions
        else:
            return all_available_actions

    def plot_board_state(
        self: "Connect4",
        board_state: Optional[npt.NDArray[np.float64]] = None,
        update: bool = False,
    ) -> None:
        if board_state is None:
            board_state = self._get_board()

        no_rows, no_cols = board_state.shape

        # set x and y values and set radius
        x = np.array(list(range(no_cols)) * no_rows)
        y = np.array([list(range(no_rows)) for _ in range(no_cols)]).flatten("F")
        r = [1000] * len(x)

        # find colors
        color_map = {1: "r", -1: "y", 0: "w"}
        colors = [color_map[v] for v in board_state.flatten()]

        # Create figure
        if self.fig is None or update is False:
            plt.ion()
            self.fig = plt.figure(figsize=(no_cols * 2, no_rows))
            self.ax = self.fig.add_subplot(111)
            self.scatter = self.ax.scatter(x + 0.5, y + 0.5, s=r, c=colors)
            self.ax.set_facecolor("blue")
            plt.xticks(ticks=np.arange(7) + 0.5, labels=list(map(str,np.arange(1, self._no_cols + 1))))
            self.ax.get_yaxis().set_visible(False)
            plt.xlim([0, no_cols])
            plt.ylim([0, no_rows])
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

        else:
            self.scatter = self.ax.scatter(x + 0.5, y + 0.5, s=r, c=colors)

        if update is False:
            plt.show()
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


@numba.njit
def update_winning_possibilities(
    board: npt.NDArray[np.float64],
    current_possibilities: npt.NDArray[np.float64],
    player: int,
    col: int,
    row: int,
) -> None:
    max_rows, max_cols = board.shape
    # eight direction possibilities
    directions = [(-1, -1), (0, -1), (1, -1), (1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0)]
    for row_dir, col_dir in directions:
        # check if direction is possible
        tmp_row = row + 2 * row_dir
        tmp_col = col + 2 * col_dir
        if not (tmp_row < max_rows and tmp_row >= 0 and tmp_col < max_cols and tmp_col >= 0):
            continue  # skip this direction

        # check if the two possibilities are possible
        tmp_row = row + 3 * row_dir
        tmp_col = col + 3 * col_dir
        if tmp_row < max_rows and tmp_row >= 0 and tmp_col < max_cols and tmp_col >= 0:
            first_possibility = True
        else:
            first_possibility = False

        tmp_row = row - 1 * row_dir
        tmp_col = col - 1 * col_dir
        if tmp_row < max_rows and tmp_row >= 0 and tmp_col < max_cols and tmp_col >= 0:
            second_possibility = True
        else:
            second_possibility = False

        if not (first_possibility or second_possibility):
            continue  # skip this direction

        # count first two "stones" in direction
        count_player = 1  # one for (row,col)
        for i in range(1, 3):
            count_player += board[row + i * row_dir, col + i * col_dir] == player

        if count_player <= 1:
            continue

        # check first possibility. (row,col) is last in direction
        if first_possibility and (count_player > 2 or board[row + 3 * row_dir, col + 3 * col_dir] == player):
            for i in range(0, 4):
                current_possibilities[row + i * row_dir, col + i * col_dir] = 1

        # check second possibility (row,col) is in the middle, but majority is in current direction
        if second_possibility and (count_player > 2 or board[row - row_dir, col - col_dir] == player):
            for i in range(-1, 3):
                current_possibilities[row + i * row_dir, col + i * col_dir] = 1


@numba.njit
def find_available_winning_actions(
    current_possibilities: npt.NDArray[np.float64],
    next_row_heights: list[int],
    no_cols: int,
    no_rows: int,
) -> npt.ArrayLike:
    winning_actions = np.zeros(no_cols)  # formatted as vector of 0 or 1 (1 being winning move)
    for col, row in enumerate(next_row_heights):
        if row < no_rows and current_possibilities[row, col] == 1:
            winning_actions[col] = 1
    return np.where(winning_actions == 1)[0]


@numba.njit
def exclude_must_avoid_actions(
    available_actions: list[int],
    current_foe_possibilities: npt.NDArray[np.float64],
    next_row_heights: list[int],
    no_rows: int,
) -> npt.ArrayLike:
    action_filter = np.ones(len(available_actions))  # formatted as vector of 0 or 1 (1 being winning move)
    for action in available_actions:
        tmp_next_foe_row_height = next_row_heights[action] + 1
        if tmp_next_foe_row_height < no_rows and current_foe_possibilities[tmp_next_foe_row_height, action] == 1:
            action_filter[action] = 0
    return available_actions[np.where(action_filter == 1)[0]]


@numba.njit
def get_available_actions_numba(board: npt.NDArray[np.float64]) -> npt.ArrayLike:
    board_top_row = board[-1, :]
    available_actions = np.where(board_top_row == 0)[0]
    return available_actions


if __name__ == "__main__":
    import time

    game = Connect4()
    game.plot_board_state()
    game._place_disc(2, 1)
    input("you ready to see some more? ")
    game.plot_board_state()
    time.sleep(3)
