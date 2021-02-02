from torch import nn
class ContrastiveModel(nn.Module):
    def __init__(self, init_base_model, repr_size):
        super(ContrastiveModel, self).__init__()
        self.base_model = init_base_model(num_classes=repr_size)
        self.base_model.fc = self.build_mlp(self.base_model.fc)

    def build_mlp(self, fc):
        in_features = fc.in_features
        hidden = nn.Linear(in_features=in_features, out_features=in_features)
        return nn.Sequential(hidden, nn.ReLU, fc)

    def forward(self, x):
        # change shape so computation is done in parallel
        x = x.view(x.shape[0]*x.shape[1], 3, 8, 112, 112)
        return self.base_model(x)

    def get_repr(self, x):
        '''
        TODO: get representation after hidden layer
        :param x:
        :return:
        '''
        return None


