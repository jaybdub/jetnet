from pydantic import BaseModel


class TorchModule(BaseModel):

    def build(self):
        raise NotImplementedError