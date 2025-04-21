

class ForgivingCfg:
    """
    Open-ended config object.
    Any attribute you don't predefine is automatically mapped to None
    so 3rd-party code never raises AttributeError.
    """
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, item):
        return None            # default for every missing field

    def log_config(self):
        """
        Return a string representation that won't crash callers.
        Feel free to format it nicely; TreeBuilder only wants *something*
        printable, not a real object.
        """
        return str(self.__dict__)
    