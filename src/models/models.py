"""
Models for the experiments.
"""
import sys
import os

sys.path.extend([
    "./",
])
import seaborn as sns

import torch.nn as nn
import torch
import torchvision.models as models

column_width_pt = 516.0
pt_to_inch = 1 / 72.27
column_width_inches = column_width_pt * pt_to_inch
aspect_ratio = 4 / 3
sns.set(style="whitegrid", font_scale=1.6, rc={"figure.figsize": (column_width_inches, column_width_inches / aspect_ratio)})

# Funzione che seleziona il modello in base all'esperimento
# TO DO: Another end-to-end col blocco unsupervised
def select_model(exp_name, input_size, num_concepts, num_classes):

    if exp_name == 'independent_concept':
        return Encoder(input_size, num_concepts)
    elif exp_name == 'independent_predictor':
        return MLP(num_concepts, num_classes)
    elif exp_name == 'independent_decoder':
        return Decoder(input_size, num_concepts)
    elif exp_name == 'sequential':
        return End2End(input_size, num_concepts, num_classes)
    elif exp_name == 'joint':
        return End2End(input_size, num_concepts, num_classes)
    else:
        raise ValueError("Invalid experiment name.")

class MLP(nn.Module):
    def __init__(self, num_concepts, num_classes, expand_dim=16):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts, expand_dim),
            nn.LeakyReLU(),
            nn.Linear(expand_dim, num_classes)
        )

    def forward(self, concepts):
        output = self.classifier(concepts)
        return output

# TO DO: insert number of unsupervised concept to be learned automatically:
# - Lasciarli completemente liberi
# - Loss di indipendenza tra i neuroni (perpendicolarità)
class Encoder(nn.Module):
    def __init__(self, input_size, num_concepts, feature_sizes=(16, 32, 64, 128)):
        super(Encoder, self).__init__()
        
        self.input_size = input_size
        self.conv_layers = nn.ModuleList()
        self.feature_sizes = feature_sizes

        in_channels = input_size[0]
        for out_channels in feature_sizes:
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        self.fc_input_features = self._calculate_fc_input_features()
        self.fc = nn.Linear(self.fc_input_features, num_concepts)
    def _calculate_fc_input_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_size[0], self.input_size[1], self.input_size[2])
            output = dummy_input
            for conv_layer in self.conv_layers:
                output = conv_layer(output)
            output_size = torch.flatten(output, start_dim=1).size(1)
        return output_size

    def forward(self, x):
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        return x

class Decoder(nn.Module):
    def __init__(self, output_size, num_concepts, feature_sizes=(16, 32, 64, 128)):
        super(Decoder, self).__init__()

        self.feature_sizes = feature_sizes[::-1]

        self.start_size = output_size[1] // (2 ** len(self.feature_sizes))
        self.fc = nn.Linear(num_concepts, self.feature_sizes[0] * self.start_size * self.start_size)

        self.convs_transpose = nn.ModuleList()
        in_channels = self.feature_sizes[0]

        for out_channels in self.feature_sizes[1:]:
            self.convs_transpose.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
                    nn.ReLU(inplace=True)
                )
            )
            in_channels = out_channels

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels, output_size[0], kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, concepts):

        x = self.fc(concepts)
        x = x.view(-1, self.feature_sizes[0], self.start_size, self.start_size)
        for conv_transpose_layer in self.convs_transpose:
            x = conv_transpose_layer(x)
        x = self.final_layer(x)

        return x



class IndependentMLP(nn.Module):
    def __init__(self, input_size, num_concepts, num_classes):

        super(IndependentMLP, self).__init__()
        self.concept_encoder = Encoder(input_size, num_concepts)
        self.predictor = MLP(num_concepts, num_classes)

    def forward(self, x):

        c = self.concept_encoder(x)
        y = self.predictor(c)

        return c, y
class End2End(nn.Module):
    def __init__(self,input_size, num_concepts, num_classes):

        super(End2End, self).__init__()
        self.concept_encoder = Encoder(input_size, num_concepts)
        self.predictor = MLP(num_concepts, num_classes)
        self.concept_decoder = Decoder(input_size, num_concepts)

    def forward(self, x):

        c = self.concept_encoder(x)

        # Classification:
        # Shape -> [0, 1, 2]

        # Regression:
        # x_pos -> [..., ....]
        # orientation -> [..., ....]


        y = self.predictor(c)

        x_tilde = self.concept_decoder(c)
        c_tilde = self.concept_encoder(x_tilde)

        return c, y, x_tilde, c_tilde

def test_encoder(img_size, num_channels, num_concepts):
    encoder = Encoder(input_size=(num_channels, img_size, img_size), num_concepts=num_concepts, feature_sizes=(16, 32, 64, 128, 256))
    #print(encoder)

    # Feed a dummy tensor.
    x = torch.randn(1, num_channels, img_size, img_size)
    concepts = encoder(x)
    print(concepts.size())

    assert concepts.size() == (1, 3)

def test_decoder(img_size, num_channels, num_concepts):

    decoder = Decoder(output_size=(num_channels, img_size, img_size), num_concepts=num_concepts, feature_sizes=(16, 32, 64, 128, 256))
    #print(decoder)

    # Feed a dummy tensor.
    concepts = torch.randn(1, 3)
    x = decoder(concepts)
    print(x.size())

    assert x.size() == (1, num_channels, img_size, img_size)

def test_MLP(num_concepts, num_classes):

    mlp = MLP(num_concepts, num_classes)
    #print(mlp)

    # Feed a dummy tensor.
    concepts = torch.randn(1, num_concepts)
    output = mlp(concepts)
    print(output.size())

    assert output.size() == (1, num_classes)

def test_endtoend(img_size, num_channels, num_concepts, num_classes):

    model = End2End(input_size=(num_channels, img_size, img_size), num_concepts=num_concepts, num_classes=num_classes)
    #print(model)

    # Feed a dummy tensor.
    x = torch.randn(1, num_channels, img_size, img_size)
    c, y, x_tilde = model(x)
    print(c.size())
    print(y.size())
    print(x_tilde.size())

# Main.
if __name__ == "__main__":

    size = 64
    channels = 1
    concepts = 3
    classes = 2

    test_encoder(size, channels, concepts)
    test_decoder(size, channels, concepts)
    test_MLP(concepts, classes)
    test_endtoend(size, channels, concepts, classes)