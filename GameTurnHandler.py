import itertools


class GameTurnHandler:
    _current_player_value: int
    _next_player_value: int
    _player_values: list[int]
    _starting_position: int
    _current_player_turn: int
    _next_player_turn: int

    def __init__(self, player_values: list[int] = [0], starting_position: int = 0):
        self.setup(player_values, starting_position)

    def setup(self, player_values: list[int], starting_position: int = 0):
        self._player_values = player_values
        self._starting_position = starting_position

        cycle = itertools.cycle(range(len(self._player_values)))
        self._player_turn_cycle = itertools.islice(cycle, starting_position, None)

        self._current_player_turn = next(self._player_turn_cycle)
        self._next_player_turn = next(self._player_turn_cycle)

        self._current_player_value = self._player_values[self._current_player_turn]
        self._next_player_value = self._player_values[self._next_player_turn]

    def get_current_player_value(self) -> int:
        return self._current_player_value

    def get_next_player_value(self) -> int:
        return self._next_player_value

    def next_turn(self) -> None:
        self._current_player_turn = self._next_player_turn
        self._current_player_value = self._next_player_value

        self._next_player_turn = next(self._player_turn_cycle)
        self._next_player_value = self._player_values[self._next_player_turn]

    def copy(self):
        new_game_handler = GameTurnHandler(
            self._player_values, self._current_player_turn
        )
        return new_game_handler


if __name__ == "__main__":
    game_turn_handler = GameTurnHandler([4, 5, 6, 7, 9], 3)
    print(
        game_turn_handler.get_current_player_value(),
        game_turn_handler.get_next_player_value(),
    )
    game_turn_handler.next_turn()
    print(
        game_turn_handler.get_current_player_value(),
        game_turn_handler.get_next_player_value(),
    )
