
import os
from absl import app

import argparse
import sys
import ast
import os
import datetime
sys.path.append(os.path.dirname(os.path.dirname(sys.path[0])))
from utilities import *
import time
import torch
from torch import nn
import numpy as np
import pickle
from torch.cuda.amp import autocast,GradScaler
from ast_models import ASTModel
from dataloader import get_data
import torch.nn.functional as F

os.environ['TORCH_HOME'] = '../ast/pretrained_models'
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--data_path',type=str, default='', help='Path to a directory containing training data')
parser.add_argument('--exp_dir', type=str, default='', help='Path to a directory expected to contain model checkpoints')
parser.add_argument('--logs_dir', type=str, default='', help='Path to a directory expected to contain model logs')
parser.add_argument('--cache_dir', type=str, default='', help='Path to a directory where model data will be cached to speed up training')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size for training')
parser.add_argument('--n_print_steps', type=int, default=8, help='How often to print loss in steps')
parser.add_argument('--n_epochs', type=int, default=20, help='Number of epochs to train over the dataset')
parser.add_argument('--dropout', type=float, default=0.5, help='Dropout Rate')
parser.add_argument('--data_percent', type=float, default=1, help='Approx percentage of input data to use for training')
parser.add_argument('--train', type=ast.literal_eval, default='False', help="Training mode")
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float, metavar='LR', help='initial learning rate')
parser.add_argument("--optim", type=str, default="adam", help="training optimizer", choices=["sgd", "adam"])


def train(audio_model, train_loader, test_loader, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on ' + str(device))
    torch.set_grad_enabled(True)

    # Initialize all of the statistics we want to keep track of
    batch_time = AverageMeter()
    per_sample_time = AverageMeter()
    data_time = AverageMeter()
    per_sample_data_time = AverageMeter()
    loss_meter = AverageMeter()
    per_sample_dnn_time = AverageMeter()
    progress = []
    # best_cum_mAP is checkpoint ensemble from the first epoch to the best epoch
    best_epoch, best_cum_epoch, best_mAP, best_acc, best_cum_mAP = 0, 0, -np.inf, -np.inf, -np.inf
    global_step, epoch = 0, 0
    start_time = time.time()
    exp_dir = args.exp_dir

    def _save_progress():
        progress.append([epoch, global_step, best_epoch, best_mAP,
                time.time() - start_time])
        with open("%s/progress.pkl" % exp_dir, "wb") as f:
            pickle.dump(progress, f)
    loss_fn = nn.CrossEntropyLoss()
    device_ids=[*range(torch.cuda.device_count())]

    args.labels_dim = audio_model.label_dim
    model_with_loss = nn.DataParallel(ModelWithLoss(audio_model, loss_fn), device_ids=device_ids)
    model_with_loss.cuda(0)
    print(model_with_loss.device_ids, model_with_loss.output_device, model_with_loss.src_device_obj)

    # Set up the optimizer
    trainables = [p for p in model_with_loss.parameters() if p.requires_grad]
    print('Total parameter number is : {:.3f} million'.format(sum(p.numel() for p in audio_model.parameters()) / 1e6))
    print('Total trainable parameter number is : {:.3f} million'.format(sum(p.numel() for p in trainables) / 1e6))
    optimizer = torch.optim.Adam(trainables, args.lr, weight_decay=5e-7, betas=(0.95, 0.999))
#    optimizer = torch.optim.SGD(trainables, args.lr)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, list(range(5,26)), gamma=0.85)
    main_metrics = 'acc'
    warmup = False
    print('now training with  main metrics: {:s}, loss function: {:s}, learning rate scheduler: {:s}'.format(str(main_metrics), str(loss_fn), str(scheduler)))
    args.loss_fn = loss_fn

    epoch += 1
    # for amp
    scaler = GradScaler()

    print("current #steps=%s, #epochs=%s" % (global_step, epoch))
    print("start training...")
    result = np.zeros([args.n_epochs, 10])
    model_with_loss.train()
    train_len = -1
    while epoch < args.n_epochs + 1:
        begin_time = time.time()
        end_time = time.time()
        model_with_loss.train()
        print('---------------')
        print(datetime.datetime.now())
        print("current #epochs=%s, #steps=%s" % (epoch, global_step))

        for i, (audio_input_dict, labels) in enumerate(train_loader):
            audio_input = torch.Tensor(audio_input_dict['audio_normalized'].numpy())
            labels = torch.Tensor(labels.numpy())
            B = audio_input.size(0)
#            audio_input = audio_input.cuda(device_ids[0])#to(device, non_blocking=True)
#            labels = labels.cuda(device_ids[0]) #to(device, non_blocking=True)

            data_time.update(time.time() - end_time)
            per_sample_data_time.update((time.time() - end_time) / audio_input.shape[0])
            dnn_start_time = time.time()

            # first several steps for warm-up
            if global_step <= 1000 and global_step % 50 == 0 and warmup == True:
                warm_lr = (global_step / 1000) * args.lr
                for param_group in optimizer.param_groups:
                    param_group['lr'] = warm_lr
                print('warm-up learning rate is {:f}'.format(optimizer.param_groups[0]['lr']))

            with autocast():
                loss, audio_output = model_with_loss(labels.long(), audio_input)
                loss = loss.sum()
            # optimization if amp is not used
            #optimizer.zero_grad()
            #loss.backward()
            #optimizer.step()

            # optimiztion if amp is used
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # record loss
            loss_meter.update(loss.item(), B)
            batch_time.update(time.time() - end_time)
            per_sample_time.update((time.time() - end_time)/audio_input.shape[0])
            per_sample_dnn_time.update((time.time() - dnn_start_time)/audio_input.shape[0])

            print_step = global_step % args.n_print_steps == 0
            early_print_step = epoch == 0 and global_step % (args.n_print_steps/10) == 0
            print_step = print_step or early_print_step

            if print_step and global_step != 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                  'Per Sample Total Time {per_sample_time.avg:.5f}\t'
                  'Per Sample Data Time {per_sample_data_time.avg:.5f}\t'
                  'Per Sample DNN Time {per_sample_dnn_time.avg:.5f}\t'
                  'Train Loss {loss_meter.avg:.4f}\t'.format(
                   epoch, i, train_len, per_sample_time=per_sample_time, per_sample_data_time=per_sample_data_time,
                      per_sample_dnn_time=per_sample_dnn_time, loss_meter=loss_meter), flush=True)
                if np.isnan(loss_meter.avg):
                    print("training diverged...")
                    return

            end_time = time.time()
            global_step += 1
        
        if train_len < 0:
            train_len = global_step * args.batch_size

        print('start validation')
        stats, valid_loss = validate(model_with_loss, device_ids, test_loader, args, epoch)

        # ensemble results
        cum_stats = validate_ensemble(args, epoch)
        cum_mAP = np.mean([stat['AP'] for stat in cum_stats])
        cum_mAUC = np.mean([stat['auc'] for stat in cum_stats])
        cum_acc = cum_stats[0]['acc']

        mAP = np.mean([stat['AP'] for stat in stats])
        mAUC = np.mean([stat['auc'] for stat in stats])
        acc = stats[0]['acc']

        middle_ps = [stat['precisions'][int(len(stat['precisions'])/2)] for stat in stats]
        middle_rs = [stat['recalls'][int(len(stat['recalls'])/2)] for stat in stats]
        average_precision = np.mean(middle_ps)
        average_recall = np.mean(middle_rs)

        if main_metrics == 'mAP':
            print("mAP: {:.6f}".format(mAP))
        else:
            print("acc: {:.6f}".format(acc))
        print("AUC: {:.6f}".format(mAUC))
        print("Avg Precision: {:.6f}".format(average_precision))
        print("Avg Recall: {:.6f}".format(average_recall))
        print("d_prime: {:.6f}".format(d_prime(mAUC)))
        print("train_loss: {:.6f}".format(loss_meter.avg))
        print("valid_loss: {:.6f}".format(valid_loss))

        if main_metrics == 'mAP':
            result[epoch-1, :] = [mAP, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_mAP, cum_mAUC, optimizer.param_groups[0]['lr']]
        else:
            result[epoch-1, :] = [acc, mAUC, average_precision, average_recall, d_prime(mAUC), loss_meter.avg, valid_loss, cum_acc, cum_mAUC, optimizer.param_groups[0]['lr']]
        np.savetxt(exp_dir + '/result.csv', result, delimiter=',')
        print('validation finished')

        if mAP > best_mAP:
            best_mAP = mAP
            if main_metrics == 'mAP':
                best_epoch = epoch

        if acc > best_acc:
            best_acc = acc
            if main_metrics == 'acc':
                best_epoch = epoch

        if cum_mAP > best_cum_mAP:
            best_cum_epoch = epoch
            best_cum_mAP = cum_mAP

        if best_epoch == epoch:
            torch.save(audio_model.state_dict(), "%s/models/best_audio_model.pth" % (exp_dir))
            torch.save(optimizer.state_dict(), "%s/models/best_optim_state.pth" % (exp_dir))

        torch.save(audio_model.state_dict(), "%s/models/audio_model.%d.pth" % (exp_dir, epoch))
        
        torch.save(optimizer.state_dict(), "%s/models/optim_state.%d.pth" % (exp_dir, epoch))

        scheduler.step()

        print('Epoch-{0} lr: {1}'.format(epoch, optimizer.param_groups[0]['lr']))

        with open(exp_dir + '/stats_' + str(epoch) +'.pickle', 'wb') as handle:
            pickle.dump(stats, handle, protocol=pickle.HIGHEST_PROTOCOL)
        _save_progress()

        finish_time = time.time()
        print('epoch {:d} training time: {:.3f}'.format(epoch, finish_time-begin_time))

        epoch += 1

        batch_time.reset()
        per_sample_time.reset()
        data_time.reset()
        per_sample_data_time.reset()
        loss_meter.reset()
        per_sample_dnn_time.reset()


def validate(audio_model, device_ids, val_loader, args, epoch):
    batch_time = AverageMeter()
    if not isinstance(audio_model, nn.DataParallel):
        if torch.cuda.is_available():
            audio_model = nn.DataParallel(audio_model, device_ids = device_ids)
    audio_model = audio_model.to(device_ids[0])

    # switch to evaluate mode
    audio_model.eval()

    end = time.time()
    A_predictions = []
    A_targets = []
    A_loss = []
    with torch.no_grad():
        for i, (audio_input_dict, labels) in enumerate(val_loader):
            audio_input = torch.Tensor(audio_input_dict['audio_normalized'].numpy())
            labels = torch.Tensor(labels.numpy())

            # compute output
            loss, audio_output = audio_model(labels.long(), audio_input)
            loss = loss.sum()
            audio_output = torch.sigmoid(audio_output)
            predictions = audio_output.to('cpu').detach().squeeze()

            A_predictions.append(predictions)
            labels_onehot = F.one_hot(labels.to(torch.int64), num_classes=args.labels_dim)
            A_targets.append(labels_onehot.squeeze())

            # compute the loss
            A_loss.append(loss.to('cpu').detach())

            batch_time.update(time.time() - end)
            end = time.time()

        audio_output = torch.cat(A_predictions)
        target = torch.cat(A_targets)
        loss = np.mean(A_loss)
        stats = calculate_stats(audio_output, target)

        # save the prediction here
        exp_dir = args.exp_dir
        if os.path.exists(exp_dir+'/predictions') == False:
            os.mkdir(exp_dir+'/predictions')
            np.savetxt(exp_dir+'/predictions/target.csv', target, delimiter=',')
        np.savetxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', audio_output, delimiter=',')

    return stats, loss

def validate_ensemble(args, epoch):
    exp_dir = args.exp_dir
    target = np.loadtxt(exp_dir+'/predictions/target.csv', delimiter=',')
    if epoch == 1:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/predictions_1.csv', delimiter=',')
    else:
        cum_predictions = np.loadtxt(exp_dir + '/predictions/cum_predictions.csv', delimiter=',') * (epoch - 1)
        predictions = np.loadtxt(exp_dir+'/predictions/predictions_' + str(epoch) + '.csv', delimiter=',')
        cum_predictions = cum_predictions + predictions
        # remove the prediction file to save storage space
        os.remove(exp_dir+'/predictions/predictions_' + str(epoch-1) + '.csv')

    cum_predictions = cum_predictions / epoch
    np.savetxt(exp_dir+'/predictions/cum_predictions.csv', cum_predictions, delimiter=',')

    stats = calculate_stats(cum_predictions, target)
    return stats

def validate_wa(audio_model, val_loader, args, start_epoch, end_epoch):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    exp_dir = args.exp_dir

    sdA = torch.load(exp_dir + '/models/audio_model.' + str(start_epoch) + '.pth', map_location=device)

    model_cnt = 1
    for epoch in range(start_epoch+1, end_epoch+1):
        sdB = torch.load(exp_dir + '/models/audio_model.' + str(epoch) + '.pth', map_location=device)
        for key in sdA:
            sdA[key] = sdA[key] + sdB[key]
        model_cnt += 1

        # if choose not to save models of epoch, remove to save space
        if args.save_model == False:
            os.remove(exp_dir + '/models/audio_model.' + str(epoch) + '.pth')

    # averaging
    for key in sdA:
        sdA[key] = sdA[key] / float(model_cnt)

    audio_model.load_state_dict(sdA)

    torch.save(audio_model.state_dict(), exp_dir + '/models/audio_model_wa.pth')
    device_ids = [*range(torch.cuda.device_cound())]
    stats, loss = validate(audio_model, device_ids, val_loader, args, 'wa')
    return stats

class ModelWithLoss(nn.Module):
    def __init__(self, model, loss):
        super().__init__()
        self.model = model
        self.loss = loss
        
    def forward(self, targets, *inputs):
        outputs = self.model(*inputs)
        loss = self.loss(outputs, targets.squeeze())
        return torch.unsqueeze(loss,0),outputs
    
def main():
    args = parser.parse_args()
    train_dataset, val_dataset, test_dataset = get_data(args.data_path, args.batch_size, args.cache_dir, args.data_percent)
    X, y = next(iter(train_dataset))
    print([(key, X[key].shape) for key in X] + [y.shape])
    input_tdim = X['audio_normalized'].shape[1]
    label_dim = 399
    test_input = torch.rand([10, input_tdim, 128])
    audio_model = ASTModel(label_dim=label_dim, input_tdim=input_tdim, imagenet_pretrain=True, audioset_pretrain=False, model_size='small224')
    # output should be in shape [10, 527], i.e., 10 samples, each with prediction of 527 classes.
    print("\nCreating experiment directory: %s" % args.exp_dir)
    
    os.makedirs("%s/models" % args.exp_dir, exist_ok=True)
    with open("%s/args.pkl" % args.exp_dir, "wb") as f:
        pickle.dump(args, f)
        
    print('Now starting training for {:d} epochs'.format(args.n_epochs))
    train(audio_model, train_dataset, val_dataset, args)

if __name__ == '__main__':
    main()

