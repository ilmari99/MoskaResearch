import logging
import random
import datetime
import os
""" Create a logger to log different events by user.
A logger is day specific. It will create a new file for each day.
"""

class SessionLogger(logging.LoggerAdapter):
    """ This logger will add the username to the log message.
    """
    def __init__(self, session, logger, extra):
        self.session = session
        super().__init__(logger, extra)

    def process(self, msg, kwargs):
        username = "Unknown"
        if self.session and 'username' in self.session:
            username = self.session['username']
        return f"[{username}]: {msg}", kwargs
    
def init_logger(session):
    """ Create a logger. This logger will not log Flask logs. Thye will go to terminal instead. 
    """
    #logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a", format="%(asctime)s-%(levelname)s-%(message)s")
    base_logger = logging.getLogger("app_logs"+str(random.randint(0,1000000)))
    if not os.path.exists("./Logs"):
        os.mkdir("./Logs")
    # Log file for every six hours
    # TODO: DOESN'T WORK!!
    log_file = "./Logs/" + "app"+datetime.datetime.now().strftime("%Y-%m-%d")+"-"+str(datetime.datetime.now().hour//6)+".log"
    fh = logging.FileHandler(log_file, mode="a")
    form = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
    fh.setFormatter(form)
    base_logger.addHandler(fh)
    base_logger.setLevel(logging.INFO)
    logger = SessionLogger(session,base_logger, {})
    return logger
