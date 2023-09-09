class Debug:
    STATUS = False

    @classmethod
    def print(cls, msg):
        if cls.STATUS:
            print(msg)

    @classmethod
    def set_status(cls, status=bool):
        cls.STATUS = status
