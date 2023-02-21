import os
import sys
# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
import sys
from Moska.Player.AbstractPlayer import AbstractPlayer
from typing import Any, Callable, Dict, Iterable, List, Tuple
from Utils import replace_setting_values
"""
This file contains the PlayerWrapper class, which is used to wrap a player class and settings into a single object.
This object is then used to create a player instance. It is done this way to ease multiprocessing, and create filenames based on the game number.
"""

class PlayerWrapper:
    """ Wraps a player class and settings into a single object.
    Avoid ugly code, which uses a tuple of (player_class, settings : Callable) everywhere.
    """
    def __init__(self, player_class: AbstractPlayer, settings: Dict[str, Any], infer_log_file = False):
        """ Settings should have '{x}' somewhere in it, which will be replaced by the game number.
        """
        if not issubclass(player_class, AbstractPlayer):
            raise TypeError(f"Player class must be a subclass of AbstractPlayer, but is {player_class}")
        if not isinstance(settings, dict):
            raise TypeError(f"Settings must be a dict, but is {settings}")
        if infer_log_file and not 'log_file' in settings:
            name = settings.get("name", player_class.__name__)
            settings["log_file"] = "Game_{x}-" + name + ".log"
        self.player_class = player_class
        self.settings = settings.copy()
        return
    
    def _get_instance_settings(self, game_id : int) -> None:
        """ Create a new settings dict, with the game id replaced.
        """
        # Create a new dict, so that the original settings are not changed.
        instance_settings = replace_setting_values(self.settings, game_id)
        return instance_settings
    
    def __repr__(self) -> str:
        return f"PlayerWrapper({self.player_class.__name__}, {self.settings})"
    
    def __call__(self, game_id : int = 0,instance_settings = None) -> AbstractPlayer:
        """ Create a player instance.
        """
        if not instance_settings:
            instance_settings = self._get_instance_settings(game_id)
        return self.player_class(**instance_settings)