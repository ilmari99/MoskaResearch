import logging

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
    
class Logger:
    """ Creates a SessionLogger, that is either created or returned if already created.
    """
    def __init__(self, session):
        self.session = session
        self.init_logger(session)

    def get_logger(self):
        return self.logger
    
    def init_logger(self,session):
        """ Create a logger. This logger will not log Flask logs. Thye will go to terminal instead. 
        """
        #logging.basicConfig(level=logging.INFO, filename="app.log", filemode="a", format="%(asctime)s-%(levelname)s-%(message)s")
        base_logger = logging.getLogger("app_logs")
        fh = logging.FileHandler("app.log", mode="a")
        form = logging.Formatter("%(asctime)s-%(levelname)s-%(message)s")
        fh.setFormatter(form)
        base_logger.addHandler(fh)
        base_logger.setLevel(logging.INFO)
        self.logger = SessionLogger(session,base_logger, {})
        return self.logger
    
    def __new__(cls, session):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Logger, cls).__new__(cls)
        return cls.instance.init_logger(session)
