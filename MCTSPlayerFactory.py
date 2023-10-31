
import ConfigHandler
from Connect4Game import Connect4
from Connect4Players import MCTSPlayer


class MCTSPlayerFactory:
    @staticmethod
    def create_player(
        game: Connect4,
        player: int,
        next_player: int,
        name: str,
        config_handler: ConfigHandler,
    ) -> MCTSPlayer:
        # get config from config_handler based on inputted name
        mctsplayer_config = ConfigHandler.MCTSPlayerConfig(config_handler, name)
        # create mctsplayer using config and inputs
        mctsplayer = MCTSPlayer(game, player, next_player, mctsplayer_config)
        return mctsplayer


class MCTSPlayerNames:
    normal: str = "normal"
    hard: str = "hard"
    god: str = "god"

if __name__ == "__main__":
    a = MCTSPlayerNames.normal
    print(a, type(a))
