from Connect4Game import Connect4
from Connect4Players import MCTSPlayer


class MCTSPlayerFactory:
    @staticmethod
    def create_normal_player(game: Connect4, player: int, next_player: int) -> MCTSPlayer:
        mctsplayer = MCTSPlayer(game, player, next_player)

        mctsplayer.max_count = 5e2
        mctsplayer.max_depth = 2

        return mctsplayer

    @staticmethod
    def create_hard_player(game: Connect4, player: int, next_player: int) -> MCTSPlayer:
        mctsplayer = MCTSPlayer(game, player, next_player)

        mctsplayer.max_count = 2e3
        mctsplayer.max_depth = 5
        mctsplayer.rave_param = 2**0

        return mctsplayer

    @staticmethod
    def create_god_player(game: Connect4, player: int, next_player: int) -> MCTSPlayer:
        mctsplayer = MCTSPlayer(game, player, next_player)

        mctsplayer.max_count = 1e4
        mctsplayer.max_depth = 100
        mctsplayer.rave_param = 2**0

        return mctsplayer
