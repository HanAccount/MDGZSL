import numpy as np
import torch
import torch.optim as optim
import glob
import json
import argparse
import os
import random
import math
from time import gmtime, strftime
from models import *
from dataset_GBU import FeatDataLayer, DATA_LOADER
# from utils import *
from .utils import *
from sklearn.metrics.pairwise import cosine_similarity
import torch.backends.cudnn as cudnn
import classifier


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='AwA2',help='dataset: CUB, AWA2, APY, FLO, SUN')
parser.add_argument('--dataroot', default='./data', help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--image_embedding', default='res101', type=str)
parser.add_argument('--class_embedding', default='att', type=str)

parser.add_argument('--gen_nepoch', type=int, default=400, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate to train generater')

parser.add_argument('--zsl', type=bool, default=False, help='Evaluate ZSL or GZSL')
parser.add_argument('--finetune', type=bool, default=False, help='Use fine-tuned feature')
parser.add_argument('--ga', type=float, default=15, help='relationNet weight')
parser.add_argument('--beta', type=float, default=1, help='tc weight')
parser.add_argument('--weight_decay', type=float, default=1e-6, help='weight_decay')
parser.add_argument('--dis', type=float, default=3, help='Discriminator weight')
parser.add_argument('--dis_step', type=float, default=2, help='Discriminator update interval')
parser.add_argument('--kl_warmup', type=float, default=0.01, help='kl warm-up for VAE')
parser.add_argument('--tc_warmup', type=float, default=0.001, help='tc warm-up')

parser.add_argument('--vae_dec_drop', type=float, default=0.5, help='dropout rate in the VAE decoder')
parser.add_argument('--vae_enc_drop', type=float, default=0.4, help='dropout rate in the VAE encoder')
parser.add_argument('--ae_drop', type=float, default=0.2, help='dropout rate in the auto-encoder')

parser.add_argument('--classifier_lr', type=float, default=0.001, help='learning rate to train softmax classifier')
parser.add_argument('--classifier_steps', type=int, default=50, help='training steps of the classifier')

parser.add_argument('--batchsize', type=int, default=64, help='input batch size')
parser.add_argument('--nSample', type=int, default=1200, help='number features to generate per class')

parser.add_argument('--disp_interval', type=int, default=200)
parser.add_argument('--save_interval', type=int, default=10000)
parser.add_argument('--evl_interval',  type=int, default=400)
parser.add_argument('--evl_start',  type=int, default=0)
parser.add_argument('--manualSeed', type=int, default=5606, help='manual seed')

parser.add_argument('--latent_dim', type=int, default=20, help='dimention of latent z')
parser.add_argument('--q_z_nn_output_dim', type=int, default=128, help='dimention of hidden layer in encoder')
parser.add_argument('--S_dim', type=int, default=1024)
parser.add_argument('--NS_dim', type=int, default=1024)

parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--center_margin', default=100, type=int, help='margin')
parser.add_argument('--center_weight', default=0.5, type=float, help='cw')
parser.add_argument('--recons_weights', default=0.001, type=float, help='rw')
parser.add_argument('--incenter_weight', default=0.5, type=float, help='rw')
parser.add_argument('--lambda1', type=float, default=10, help='gradient penalty regularizer, following WGAN-GP')
# parser.add_argument('--gpu', default='0', type=str, help='index of GPU to use')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--critic_iter', type=int, default=5, help='critic iteration, following WGAN-GP')
parser.add_argument('--i_c', type=float, default=0.1, help='information constrain')
parser.add_argument('--ntrain_class', type=int, default=150, help='number of seen classes')
parser.add_argument('--nclass_seen', type=int, default=150, help='number of seen classes')
parser.add_argument('--attSize', type=int, default=85)
parser.add_argument('--ngh', type=int, default=1024)
parser.add_argument('--CS_dim', type=int, default=20)
parser.add_argument('--CNS_dim', type=int, default=20)



opt = parser.parse_args()

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
np.random.seed(opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True
print('Running parameters:')
print(json.dumps(vars(opt), indent=4, separators=(',', ': ')))
opt.gpu = torch.device("cuda:"+opt.gpu if torch.cuda.is_available() else "cpu")


def train():
    dataset = DATA_LOADER(opt)
    opt.C_dim = dataset.att_dim
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class
    out_dir = 'Dual_out/{}/b-{}_g-{}_lr-{}_sd-{}_dis-{}_nS-{}_nZ-{}_bs-{}_step-{}_ae-drop-{}_tc-warmup-{}_cls-lr-{}_nCS-{}'.format(opt.dataset,
                    opt.beta, opt.ga, opt.lr, opt.S_dim, opt.dis, opt.nSample, opt.Z_dim, opt.batchsize, opt.dis_step,
                    opt.ae_drop, opt.tc_warmup, opt.classifier_lr, opt.CS_dim)
    os.makedirs(out_dir, exist_ok=True)
    print("The output dictionary is {}".format(out_dir))

    log_dir = out_dir + '/log_{}.txt'.format(opt.dataset)
    with open(log_dir, 'w') as f:
        f.write('Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    dataset.feature_dim = dataset.train_feature.shape[1]
    opt.X_dim = dataset.feature_dim
    opt.Z_dim = opt.latent_dim
    opt.y_dim = dataset.ntrain_class

    data_layer = FeatDataLayer(dataset.train_label.numpy(), dataset.train_feature.cpu().numpy(), opt)

    opt.niter = int(dataset.ntrain/opt.batchsize) * opt.gen_nepoch

    result_gzsl_soft = Result()
    result_zsl_soft = Result()

    model = VAE(opt).to(opt.gpu)
    relationNet = RelationNet(opt).to(opt.gpu)
    discriminator = Discriminator(opt).to(opt.gpu)
    ae = AE(opt).to(opt.gpu)
    ae_att = AE_att(opt).to(opt.gpu)
    # print(model)

    with open(log_dir, 'a') as f:
        f.write('\n')
        f.write('Generative Model Training Start:')
        f.write(strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime()) + '\n')

    start_step = 0
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    relation_optimizer = optim.Adam(relationNet.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    dis_optimizer = optim.Adam(discriminator.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_optimizer = optim.Adam(ae.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ae_att_optimizer = optim.Adam(ae_att.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
    ones = torch.ones(opt.batchsize, dtype=torch.long, device=opt.gpu)
    zeros = torch.zeros(opt.batchsize, dtype=torch.long, device=opt.gpu)
    mse = nn.MSELoss().to(opt.gpu)


    iters = math.ceil(dataset.ntrain/opt.batchsize)
    beta = 0.01
    coin = 0
    gamma = 0
    for it in range(start_step, opt.niter+1):

        if it % iters == 0:
            beta = min(opt.kl_warmup*(it/iters), 1)
            gamma = min(opt.tc_warmup * (it / iters), 1)

        blobs = data_layer.forward()
        feat_data = blobs['data']
        labels_numpy = blobs['labels'].astype(int)
        labels = torch.from_numpy(labels_numpy.astype('int')).to(opt.gpu)
        # print(labels)
        C = np.array([dataset.train_att[i,:] for i in labels])
        C = torch.from_numpy(C.astype('float32')).to(opt.gpu)
        # [64,85]
        # print(C.shape)
        X = torch.from_numpy(feat_data).to(opt.gpu)
        sample_C = torch.from_numpy(np.array([dataset.train_att[i, :] for i in labels.unique()])).to(opt.gpu)
        sample_C_n = labels.unique().shape[0]
        sample_label = labels.unique().cpu()
        # [40,85]
        # print(dataset.train_att.shape)
        x_mean, z_mu, z_var, z = model(X, C)
        loss, ce, kl = multinomial_loss_function(x_mean, X, z_mu, z_var, z, beta=beta)

        sample_labels = np.array(sample_label)
        re_batch_labels = []
        for label in labels_numpy:
            index = np.argwhere(sample_labels == label)
            re_batch_labels.append(index[0][0])
        re_batch_labels = torch.LongTensor(re_batch_labels)
        one_hot_labels = torch.zeros(opt.batchsize, sample_C_n).scatter_(1, re_batch_labels.view(-1, 1), 1).to(opt.gpu)

        x1, h1, hs1, hn1 = ae(x_mean)

        c1, latent, cs1, cns1 = ae_att(C)
        att_rec = mse(C, c1)

        x11, h11, hs11, hn11 = ae(x_mean, cs1)
        c11, latent1, cs11, cns11 = ae_att(C, h1)
        cross_recons_loss = mse(x11, X) + mse(C, c11)

        relations = relationNet(hs1, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])
        p_loss = opt.ga * mse(relations, one_hot_labels)
        ring_loss = ring_loss_minimizer(X, x_mean)

        x2, h2, hs2, hn2 = ae(X)
        x22, h22, hs22, hn22 = ae(X, cs1)
        c12, latent12, cs12, cns12 = ae_att(C, h2)
        cross_recons_loss = cross_recons_loss + mse(X, x22) + mse(C, c12)

        relations = relationNet(hs2, sample_C)
        relations = relations.view(-1, labels.unique().cpu().shape[0])

        p_loss = p_loss + opt.ga * mse(relations, one_hot_labels)

        rec = mse(x1, X) + mse(x2, X)
        total_rec = att_rec + rec + cross_recons_loss

        if coin > 0:
            s_score = discriminator(h1)

            c_score = discriminator(latent, types='att')
            att_tc_loss = opt.beta * gamma * ((c_score[:, :1] - c_score[:, 1:]).mean())

            tc_loss = opt.beta * gamma * ((s_score[:, :1] - s_score[:, 1:]).mean())
            s_score = discriminator(h2)
            tc_loss = tc_loss + opt.beta * gamma * ((s_score[:, :1] - s_score[:, 1:]).mean())

            loss = loss + p_loss + total_rec + tc_loss + att_tc_loss + ring_loss
            coin -= 1
        else:
            s, n = permute_dims(hs1, hn1)

            cs, cns = permute_dims(cs1, cns1)
            k = torch.cat((cs, cns), 1).detach()
            c_score = discriminator(latent, types='att')
            k_score = discriminator(k, types='att')
            att_tc_loss = opt.dis * (F.cross_entropy(c_score, zeros) + F.cross_entropy(k_score, ones))

            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h1)
            n_score = discriminator(b)
            tc_loss = opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            s, n = permute_dims(hs2, hn2)
            b = torch.cat((s, n), 1).detach()
            s_score = discriminator(h2)
            n_score = discriminator(b)
            tc_loss = tc_loss + opt.dis * (F.cross_entropy(s_score, zeros) + F.cross_entropy(n_score, ones))

            tc_loss = tc_loss + att_tc_loss
            dis_optimizer.zero_grad()
            tc_loss.backward(retain_graph=True)
            dis_optimizer.step()

            loss = loss + p_loss + total_rec + ring_loss
            coin += opt.dis_step
        optimizer.zero_grad()
        relation_optimizer.zero_grad()
        ae_optimizer.zero_grad()
        ae_att_optimizer.zero_grad()

        loss.backward()

        optimizer.step()
        relation_optimizer.step()
        ae_optimizer.step()
        ae_att_optimizer.step()

        if it % opt.disp_interval == 0 and it:
            log_text = 'Iter-[{}/{}]; loss: {:.3f}; kl:{:.3f}; p_loss:{:.3f}; total_rec:{:.3f}; cross_recons_loss:{:.3f}; tc:{:.3f}; att_tc:{:.3f}; ring_loss:{:.3f}; gamma:{:.3f};'.format(it,
                        opt.niter, loss.item(),kl.item(),p_loss.item(), total_rec.item(), cross_recons_loss.item(), tc_loss.item(),att_tc_loss.item(), ring_loss.item(), gamma)
            log_print(log_text, log_dir)

        if it % opt.evl_interval == 0 and it > opt.evl_start:
            model.eval()
            ae.eval()
            ae_att.eval()
            gen_feat, gen_label = synthesize_feature_test(model, ae, dataset, opt)
            with torch.no_grad():
                # print(dataset.train_feature.device)
                # print(dataset.test_att.device)
                train_feature = ae.encoder(dataset.train_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_unseen_feature = ae.encoder(dataset.test_unseen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()
                test_seen_feature = ae.encoder(dataset.test_seen_feature.to(opt.gpu))[:,:opt.S_dim].cpu()

                train_att = ae_att.encoder(torch.tensor(dataset.attribute).cuda())[:, :opt.CS_dim].cpu()
                test_unseen_att = ae_att.encoder(torch.tensor(dataset.test_att).cuda())[:, :opt.CS_dim].cpu()
                test_seen_att = ae_att.encoder(torch.tensor(dataset.train_att).cuda())[:, :opt.CS_dim].cpu()

                # train_C = ae_att.encoder(C)[:, :opt.CS_dim].cpu()

            # test_seen_feature = torch.cat((test_seen_feature, test_seen_att), 0)
            # test_seen_feature = test_seen_feature + test_seen_att
            # test_unseen_feature = torch.cat((test_unseen_feature, test_unseen_att), 0)
            # test_unseen_feature = test_unseen_feature + test_unseen_att

            train_X = torch.cat((train_feature, gen_feat), 0)
            # train_X = torch.cat((train_X, train_att), dim=0)
            # print(train_X.shape)
            # train_X = train_X + train_att
            nus = opt.S_dim
            train_att = train_att.resize_((train_X.shape[0], nus))
            test_unseen_att = test_unseen_att.resize_((test_unseen_feature.shape[0], nus))
            test_seen_att = test_seen_att.resize_((test_seen_feature.shape[0], nus))
            # print(train_att.shape)
            train_Y = torch.cat((dataset.train_label, gen_label + dataset.ntrain_class), 0)
            # print(train_Y.shape)
            train_X = train_X + train_att

            test_seen_feature = test_seen_feature + test_seen_att

            test_unseen_feature = test_unseen_feature + test_unseen_att
            # print(train_X.shape)
            # print(train_Y.shape)
            # print(test_unseen_feature.shape)
            # print(test_seen_feature.shape)

            test_unseen_att1 = test_unseen_att.resize_((gen_feat.shape[0], gen_feat.shape[1]))

            test_X = gen_feat + test_unseen_att1
            # print(test_X.shape)
            if opt.zsl:
                """ZSL"""
                cls = classifier.CLASSIFIER(opt, test_X, gen_label, dataset, test_seen_feature, test_unseen_feature,
                                            dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5, 20,
                                            opt.nSample, False)
                result_zsl_soft.update(it, cls.acc)
                log_print("ZSL Softmax:", log_dir)
                log_print("Acc {:.2f}%  Best_acc [{:.2f}% | Iter-{}]".format(
                    cls.acc, result_zsl_soft.best_acc, result_zsl_soft.best_iter), log_dir)

            else:
                """ GZSL"""
                cls = classifier.CLASSIFIER(opt, train_X, train_Y, dataset, test_seen_feature, test_unseen_feature,
                                    dataset.ntrain_class + dataset.ntest_class, True, opt.classifier_lr, 0.5,
                                    opt.classifier_steps, opt.nSample, True)

                result_gzsl_soft.update_gzsl(it, cls.acc_seen, cls.acc_unseen, cls.H)

                log_print("GZSL Softmax:", log_dir)
                log_print("U->T {:.2f}%  S->T {:.2f}%  H {:.2f}%  Best_H [{:.2f}% {:.2f}% {:.2f}% | Iter-{}]".format(
                    cls.acc_unseen, cls.acc_seen, cls.H,  result_gzsl_soft.best_acc_U_T, result_gzsl_soft.best_acc_S_T,
                    result_gzsl_soft.best_acc, result_gzsl_soft.best_iter), log_dir)

                if result_gzsl_soft.save_model:
                    files2remove = glob.glob(out_dir + '/Best_model_GZSL_*')
                    for _i in files2remove:
                        os.remove(_i)
                    save_model(it, model, opt.manualSeed, log_text,
                               out_dir + '/Best_model_GZSL_H_{:.2f}_S_{:.2f}_U_{:.2f}.tar'.format(result_gzsl_soft.best_acc,
                                                                                                 result_gzsl_soft.best_acc_S_T,
                                                                                                 result_gzsl_soft.best_acc_U_T))
            ###############################################################################################################

            model.train()
            ae.train()
            ae_att.train()
        if it % opt.save_interval == 0 and it:
            save_model(it, model, opt.manualSeed, log_text,
                       out_dir + '/Iter_{:d}.tar'.format(it))
            print('Save model to ' + out_dir + '/Iter_{:d}.tar'.format(it))


if __name__ == "__main__":
    train()
