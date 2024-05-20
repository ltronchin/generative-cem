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
def select_model(exp_name, input_size, num_concepts, num_model_concepts, num_classes):

    if exp_name == 'independent_concept':
        return Encoder(input_size, num_concepts, num_embed_for_concept=8, num_model_concepts=num_model_concepts)
    elif exp_name == 'independent_predictor':
        return MLP(num_concepts, num_classes)
    elif exp_name == 'independent_decoder':
        return Decoder(input_size, num_concepts, num_model_concepts=num_model_concepts)
    elif exp_name == 'sequential':
        return End2End(input_size, num_concepts, num_classes, num_embed_for_concept=8, num_model_concepts=2)
    elif exp_name == 'joint':
        return End2End(input_size, num_concepts, num_classes, num_embed_for_concept=8, num_model_concepts=2)
    else:
        raise ValueError("Invalid experiment name.")

class MLP(nn.Module):
    def __init__(self, num_concepts, num_classes, num_model_concepts = 0, expand_dim=16):
        super(MLP, self).__init__()

        self.classifier = nn.Sequential(
            nn.Linear(num_concepts + num_model_concepts, expand_dim),
            nn.LeakyReLU(),
            nn.Linear(expand_dim, num_classes)
        )

    def forward(self, concepts):
        output = self.classifier(concepts)
        return output
#%%
# TO DO: - Loss di indipendenza tra i neuroni (perpendicolarità)
class Encoder(nn.Module):
    def __init__(self, input_size, num_concepts, num_embed_for_concept=16, num_model_concepts = 0, feature_sizes=(16, 32, 64, 128)):
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
        # self.fc = nn.Linear(self.fc_input_features, num_concepts)
        self.fc_adjust = nn.Linear(self.fc_input_features, (num_concepts + num_model_concepts) * num_embed_for_concept)
        
        self.num_concepts = num_concepts
        self.num_model_concepts = num_model_concepts
        self.num_embed_for_concept = num_embed_for_concept
        
        self.linear_layers = nn.ModuleList()
        num_linear_layers = num_concepts + num_model_concepts      
        
        for _ in range(num_linear_layers):
            self.linear_layers.append(nn.Linear(num_embed_for_concept, 1))

    def _calculate_fc_input_features(self):
        with torch.no_grad():
            dummy_input = torch.zeros(1, self.input_size[0], self.input_size[1], self.input_size[2])
            output = dummy_input
            for conv_layer in self.conv_layers:
                output = conv_layer(output)
            output_size = torch.flatten(output, start_dim=1).size(1)
        return output_size

    def forward(self, x):
        # Forward pass through convolutional layers
        for conv_layer in self.conv_layers:
            x = conv_layer(x)
        # Flatten the output
        x = torch.flatten(x, start_dim=1)
        
        x = self.fc_adjust(x)

   # Split flattened features into chunks of num_embed_for_concept and pass through linear layers
        concept_outputs = []
        model_concept_outputs = []

        for i, linear_layer in enumerate(self.linear_layers):
            chunk = x[:, i * self.num_embed_for_concept : (i + 1) * self.num_embed_for_concept]
            if i < self.num_concepts:
                # Predict a single continuous value (regression) for the first num_concepts linear layers
                concept_output = linear_layer(chunk)
                concept_outputs.append(concept_output)
            else:
                # Learn unsupervised concepts for the remaining linear layers 
                model_concept_output = linear_layer(chunk)
                model_concept_outputs.append(model_concept_output)

        # Concatenate outputs along the feature dimension
        if concept_outputs:
            concept_outputs = torch.cat(concept_outputs, dim=1)
        if self.num_model_concepts != 0:
            model_concept_outputs = torch.cat(model_concept_outputs, dim=1)
        else:
            model_concept_output = None

        return concept_outputs, model_concept_outputs

#%%
class Decoder(nn.Module):
    def __init__(self, output_size, num_concepts, num_model_concepts = 0, feature_sizes=(16, 32, 64, 128)):
        super(Decoder, self).__init__()

        self.feature_sizes = feature_sizes[::-1]

        self.start_size = output_size[1] // (2 ** len(self.feature_sizes))
        self.fc = nn.Linear(num_concepts + num_model_concepts, self.feature_sizes[0] * self.start_size * self.start_size)

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
    def __init__(self, input_size, num_concepts, num_classes, num_model_concepts=0):

        super(IndependentMLP, self).__init__()
        self.concept_encoder = Encoder(input_size, num_concepts, num_model_concepts=num_model_concepts)
        self.predictor = MLP(num_concepts + num_model_concepts, num_classes)

    def forward(self, x):
        #TODO: This is working on BOTH supervised and unsupervised!
        c = self.concept_encoder(x)
        y = self.predictor(c)

        return c, y
class End2End(nn.Module):
    def __init__(self,input_size, num_concepts, num_classes, num_embed_for_concept=8, num_model_concepts=0):

        super(End2End, self).__init__()
        self.concept_encoder = Encoder(input_size, num_concepts, num_embed_for_concept, num_model_concepts)
        self.predictor = MLP(num_concepts + num_model_concepts, num_classes)
        self.concept_decoder = Decoder(input_size, num_concepts, num_model_concepts)

    def forward(self, x):

        c1, c2 = self.concept_encoder(x)
        # print(c1.size(), c2.size())
        c = torch.cat((c1,c2), dim=1)
        # print(c.size())
        # Classification:
        # Shape -> [0, 1, 2]

        # Regression:
        # x_pos -> [..., ....]
        # orientation -> [..., ....]


        y = self.predictor(c)

        x_tilde = self.concept_decoder(c)
        c_tilde, _ = self.concept_encoder(x_tilde)

        return c1, c2, y, x_tilde, c_tilde

#%%
def test_encoder(img_size, num_channels, num_concepts, num_model_concepts=0):
    encoder = Encoder(input_size=(num_channels, img_size, img_size), num_concepts=num_concepts, 
                      num_model_concepts=num_model_concepts, feature_sizes=(16, 32, 64, 128, 256))
    print(encoder)

    # Feed a dummy tensor.
    x = torch.randn(1, num_channels, img_size, img_size)
    c1, c2 = encoder(x)
    if num_model_concepts != 0:
        print((c1.size()), c2.size())
    else:
        print(c1.size(), c2)

    return c1, c2
    # assert c1.size() == torch.Size([1,num_concepts]), c2.size() == torch.Size([1, num_model_concepts])

def test_decoder(img_size, num_channels, num_concepts, num_model_concepts = 0):

    decoder = Decoder(output_size=(num_channels, img_size, img_size), num_concepts=num_concepts, 
                      num_model_concepts=num_model_concepts, feature_sizes=(16, 32, 64, 128, 256))
    #print(decoder)

    # Feed a dummy tensor.
    concepts = torch.randn(1, num_concepts + num_model_concepts)
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

def test_endtoend(img_size, num_channels, num_concepts, num_classes, num_model_concepts = 0):

    model = End2End(input_size=(num_channels, img_size, img_size), num_concepts=num_concepts, 
                    num_classes=num_classes, num_model_concepts=num_model_concepts)
    #print(model)

    # Feed a dummy tensor.
    x = torch.randn(1, num_channels, img_size, img_size)
    c1, c2, y, x_tilde, c_tilde = model(x)
    print(c1.size(), c2.size())
    print(y.size())
    print(x_tilde.size())
    print(c_tilde.size())

# Main.
if __name__ == "__main__":

    size = 64
    channels = 1
    concepts = 3
    model_concepts = 1
    classes = 2

    test_encoder(size, channels, concepts, num_model_concepts=model_concepts)
    test_decoder(size, channels, concepts, num_model_concepts=model_concepts)
    test_MLP(concepts + model_concepts, classes)
    test_endtoend(size, channels, concepts, classes, model_concepts)
    
    #TO DO: Change training to take care of num_model_concepts
    #TO DO: Calcolare MSE tra unsupervised concepts and the CONCEPTS previously taken out from the dsprite
    #TO DO: Create new Yaml file for the experiments
