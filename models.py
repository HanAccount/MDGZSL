from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.autograd import Function
from torchvision.models import resnet101
from classifier import CLASSIFIER

class VAE(nn.Module):
    def __init__(self, args):
        super(VAE, self).__init__()
        self.z_size = args.Z_dim
        self.input_size = args.S_dim
        self.args = args
        self.q_z_nn_output_dim = args.q_z_nn_output_dim
        self.q_z_nn, self.q_z_mean, self.q_z_var = self.create_encoder()
        self.p_x_nn, self.p_x_mean = self.create_decoder()
        self.FloatTensor = torch.FloatTensor
        self.xi_bn = nn.BatchNorm1d(2048)
        self.classifier = Classifier(args)
        self.fc1 = nn.Linear(self.z_size, self.z_size)
    def create_encoder(self):
        q_z_nn = nn.Sequential(
            nn.Linear(self.args.X_dim + self.args.C_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.Dropout(self.args.vae_enc_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, self.q_z_nn_output_dim)
        )
        q_z_mean = nn.Linear(self.q_z_nn_output_dim, self.z_size)

        q_z_var = nn.Sequential(
            nn.Linear(self.q_z_nn_output_dim, self.z_size),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        return q_z_nn, q_z_mean, q_z_var


    def create_decoder(self):
        p_x_nn = nn.Sequential(
            nn.Linear(self.z_size + self.args.C_dim, 2048),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.Linear(2048, 2048),
            nn.BatchNorm1d(2048, 0.8),
            nn.Dropout(self.args.vae_dec_drop),
            nn.LeakyReLU(0.2, inplace=True)
        )
        p_x_mean = nn.Sequential(
            nn.Linear(2048, self.args.X_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )
        return p_x_nn, p_x_mean

    def reparameterize(self, mu, var):
        std = var.sqrt()
        eps = self.FloatTensor(std.size()).normal_().to(self.args.gpu)
        eps = Variable(eps)
        z = eps.mul(std).add_(mu)
        return z

    def encode(self, x, c):
        input = torch.cat((x,c),1)
        h = self.q_z_nn(input)
        h = h.view(h.size(0), -1)
        mean = self.q_z_mean(h)
        var = self.q_z_var(h)
        return mean, var

    def decode(self, z, c):
        input = torch.cat((z, c), 1)
        h = self.p_x_nn(input)
        x_mean = self.p_x_mean(h)
        return x_mean

    def forward(self, x, c, weights=None):
        z_mu, z_var = self.encode(x, c)
        z = self.reparameterize(z_mu, z_var)
        x_mean = self.decode(z, c)
        return x_mean, z_mu, z_var, z

class Discriminator_D1(nn.Module):
    def __init__(self, args):
        super(Discriminator_D1, self).__init__()
        self.args = args
        self.main = nn.Sequential(
                nn.Linear(self.args.X_dim + self.args.C_dim, self.args.ndh),
                nn.LeakyReLU(0.2, True),
                # nn.Dropout(0.2),
                nn.Linear(self.args.ndh, 1),
            )
    def forward(self, x, att):
        h = torch.cat((x, att), 1)
        h = self.main(h)
        return h
class Classifier(nn.Module):
    def __init__(self,args):
        super(Classifier, self).__init__()
        self.cls = nn.Linear(args.X_dim, args.ntrain_class) #FLO 82
        self.logic = nn.LogSoftmax(dim=1)

    def forward(self, s):
        return self.logic(self.cls(s))


class AE(nn.Module):
    def __init__(self, args):
        super(AE, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.X_dim, args.S_dim + args.NS_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.S_dim + args.NS_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )
        self.decoder_ = nn.Sequential(
            nn.Linear(args.S_dim + args.NS_dim + args.CS_dim, 2048),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.X_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(args.ae_drop),
        )

    def forward(self, x, c=None):
        z = self.encoder(x)
        s = z[:, :self.args.S_dim]
        ns = z[:, self.args.S_dim:]
        if c is not None:
            z = torch.cat((z, c), 1)
            x1 = self.decoder_(z)
        else:
            x1 = self.decoder(z)
        return x1, z, s, ns

class AE_att(nn.Module):
    def __init__(self, args):
        super(AE_att, self).__init__()
        self.args = args
        self.encoder = nn.Sequential(
            nn.Linear(args.C_dim, args.CS_dim + args.CNS_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder = nn.Sequential(
            nn.Linear(args.CS_dim + args.CNS_dim, 2048),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.C_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(args.ae_drop)
        )
        self.decoder_ = nn.Sequential(
            nn.Linear(args.CS_dim + args.CNS_dim + args.S_dim + args.NS_dim, 2048),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(args.ae_drop),
            nn.Linear(2048, args.C_dim),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(args.ae_drop)
        )


    def forward(self, c, x=None):
        h = self.encoder(c)
        cs = h[:, :self.args.CS_dim]
        cns = h[:, self.args.CS_dim:]

        if x is not None:
            h = torch.cat((h, x), 1)
            h1 = self.decoder_(h)
        else:
            h1 = self.decoder(h)

        return h1, h, cs, cns

class RelationNet(nn.Module):
    def __init__(self, args):
        super(RelationNet, self).__init__()
        self.fc1 = nn.Linear(args.C_dim + args.S_dim, 2048)
        self.fc2 = nn.Linear(2048, 1)

        self.fc3 = nn.Linear(args.CS_dim + args.S_dim, 2048)


    def forward(self, s, c, types=None):
        # c:[64,20]
        # s:[64,1024]
        c_ext = c.unsqueeze(0).repeat(s.shape[0], 1, 1)
        # c_ext:[64,64,20]
        cls_num = c_ext.shape[1]
        # cls_num:64
        s_ext = torch.transpose(s.unsqueeze(0).repeat(cls_num, 1, 1), 0, 1)
        # s_ext:[64,64,1024]
        relation_pairs = torch.cat((s_ext, c_ext), 2).view(-1, c.shape[1] + s.shape[1])
        # relation_pairs:[4096,1044]
        if types is not None:
            relation = nn.ReLU()(self.fc3(relation_pairs))
            relation = nn.Sigmoid()(self.fc2(relation))
        else:

            relation = nn.ReLU()(self.fc1(relation_pairs))
            relation = nn.Sigmoid()(self.fc2(relation))
        return relation


class Discriminator(nn.Module):
    def __init__(self, args):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(args.S_dim*2, 2)
        self.fc2 = nn.Linear(args.CS_dim*2, 2)

    def forward(self, s, types=None):
        if types == 'att':
            score = self.fc2(s)
        else:
            score = self.fc1(s)
        return nn.Sigmoid()(score)
