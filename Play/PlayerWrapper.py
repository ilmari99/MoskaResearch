import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import sys
from Moska.Player.AbstractPlayer import AbstractPlayer
from typing import Any, Callable, Dict, Iterable, List, Tuple
"""
This file contains the PlayerWrapper class, which is used to wrap a player class and settings into a single object.
This object is then used to create a player instance. It is done this way to ease multiprocessing, and create filenames based on the game number.
"""

class PlayerWrapper:
    """ Wraps a player class and settings into a single object.
    Avoid ugly code, which uses a tuple of (player_class, settings : Callable) everywhere.
    """
    def __init__(self, player_class: AbstractPlayer, settings: Dict[str, Any]):
        """ Settings should have '{x}' somewhere in it, which will be replaced by the game number.
        """
        if not issubclass(player_class, AbstractPlayer):
            raise TypeError(f"Player class must be a subclass of AbstractPlayer, but is {player_class}")
        if not isinstance(settings, dict):
            raise TypeError(f"Settings must be a dict, but is {settings}")
        
        self.player_class = player_class
        self.settings = settings.copy()
        return
    
    def _replace_game_id(self, game_id: int) -> None:
        """ Replace the '{x}' in the settings with the game id.
        """
        for key in self.settings.keys():
            if isinstance(self.settings[key], str):
                self.settings[key] = self.settings[key].format(game_id)
        return
    
    def __call__(self, game_id : int = 0) -> AbstractPlayer:
        """ Create a player instance.
        """
        self._replace_game_id(game_id)
        return self.player_class(**self.settings)