
class EarlyStoppingException(Exception):
    def __init__(self, message, epoch_stats_list=None):
        super().__init__(message)
        self.epoch_stats_list = epoch_stats_list
        self.message = message