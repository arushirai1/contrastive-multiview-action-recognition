from torch import nn
class ContrastiveModel(nn.Module):
    def __init__(self, init_base_model, repr_size):
        super(ContrastiveModel, self).__init__()
        self.base_model = init_base_model(num_classes=repr_size)
        self.base_model.fc = self.build_mlp(self.base_model.fc)
        self.eval_mode = False

    def eval_finetune(self, finetune=False, endpoint='C', num_classes=60):
        self.eval_mode=True
        self.finetune = finetune

        # copy layers from MLP depending on endpoint
        layers = []
        if endpoint == 'A':
            in_features = self.base_model.fc.in_features
        elif endpoint == 'B':
            layers.append(self.base_model.fc[0])
            in_features = self.base_model.fc[0].out_features
        elif endpoint == 'C':
            layers.append(self.base_model.fc[1])  # ReLU
            layers.append(self.base_model.fc[2])
            in_features = self.base_model.fc[2].out_features

        # add a final classifying layer
        layers.append(nn.Linear(in_features, num_classes))

        if not finetune:
            # if linear probe - freeze base_model layers
            self.base_model.requires_grad = False
        self.classifier = nn.Sequential(*layers)
        self.classifier.requires_grad = True

    def build_mlp(self, fc):
        in_features = fc.in_features
        hidden = nn.Linear(in_features=in_features, out_features=in_features)
        return nn.Sequential(hidden, nn.ReLU(), fc)

    def forward(self, x):
        x = self.base_model.extract_representation(x)
        if self.eval_mode == True:
            x = self.classifier(x)
        else:
            # when training
            x = self.base_model.fc(x)
        return x


