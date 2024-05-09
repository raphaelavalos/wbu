

class Worker:
    def train(self) -> dict:
        raise NotImplementedError

    def checkpoint(self, checkpoint_dir: str):
        raise NotImplementedError

    def restore(self, checkpoint_dir: str):
        raise NotImplementedError

