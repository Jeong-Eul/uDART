import torch.nn as nn
from functions import ReverseLayerF


class DANN(nn.Module):

    def __init__(self, dim):
        super(DANN, self).__init__()
        
        '''
        # feature extractor
        self.feature = nn.Sequential()
        self.feature.add_module('f_fc1', nn.Linear(79,128))
        self.feature.add_module('f_bn1', nn.BatchNorm1d(128))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_fc2', nn.Linear(128,100))
        self.feature.add_module('f_bn2', nn.BatchNorm1d(100))
        self.feature.add_module('f_drop1', nn.Dropout1d())
        self.feature.add_module('f_relu2', nn.ReLU(True))
        '''
        
        # label predictor
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(dim, 8))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(8))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(16, 8))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(8))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(8, 2))

        # domain classifier
        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(dim, 10))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(10))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(10, 2))

    def forward(self, input_data, alpha):
        #feature = self.feature(input_data)
        reverse_feature = ReverseLayerF.apply(input_data, alpha)
        class_output = self.class_classifier(input_data)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output
