
class BaseModel(nn.Module):
    def __init__(self, config):
        super(BaseModel, self).__init__()
        self.model = self._initialize_model(config)

    def _initialize_model(self, config):
        # Initialize a model from needed library based on config
        return

    def forward(self, x):
        return self.model(x)


class ModelFactory:
    def __init__(self, config):
        self.config = config

    def get_model(self):
        return BaseModel(self.config)
