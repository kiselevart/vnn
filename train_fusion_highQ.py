import timeit
import os
import glob
from tqdm import tqdm
import cv2
import torch
import wandb
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
import numpy as np

from dataloaders.dataset import VideoDataset
from vnn.network.video import vnn_rgb_of_highQv2
from joblib import Parallel, delayed

from vnn.network.video import vnn_fusion_highQv2


def flow(X, Ht, Wd, of_skip=1, polar=False):

    X_of = np.zeros([int(X.shape[0]/of_skip), Ht, Wd, 2])

    of_ctr = -1
    for j in range(0, X.shape[0]-of_skip, of_skip):
        of_ctr += 1
        flow = cv2.normalize(
            cv2.calcOpticalFlowFarneback(
                cv2.cvtColor(np.array(X[j+of_skip,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                cv2.cvtColor(np.array(X[j,:,:,:], dtype=np.uint8), cv2.COLOR_BGR2GRAY),
                None, 0.5, 3, 15, 3, 5, 1.2, 0
            ),
            None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F
        )
        if polar:
            mag, ang = cv2.cartToPolar(flow[:,:,0], flow[:,:,1])
            X_of[of_ctr,:,:,:] = np.concatenate([np.expand_dims(mag, axis=2), np.expand_dims(ang, axis=2)], axis=2)
        else:
            X_of[of_ctr,:,:,:] = flow

    return X_of


def compute_optical_flow(X, Ht, Wd, num_proc=4, of_skip=1, polar=False):
    X = (X.permute(0, 2, 3, 4, 1)).detach().cpu().numpy()
    optical_flow = Parallel(n_jobs=num_proc)(delayed(flow)(X[i], Ht, Wd, of_skip, polar) for i in range(X.shape[0]))
    X_of = torch.tensor(np.asarray(optical_flow)).float()
    return X_of.permute(0, 4, 1, 2, 3)


# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

nEpochs = 100
resume_epoch = 0
useTest = True
nTestInterval = 10
snapshot = 5
lr = 1e-4

dataset = 'ucf101'

if dataset == 'hmdb51':
    num_classes = 51
elif dataset == 'ucf101':
    num_classes = 101
else:
    print('We only implemented hmdb and ucf datasets.')
    raise NotImplementedError

save_dir_root = os.path.join(os.path.dirname(os.path.abspath(__file__)))
exp_name = os.path.dirname(os.path.abspath(__file__)).split('/')[-1]

if resume_epoch != 0:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) if runs else 0
else:
    runs = sorted(glob.glob(os.path.join(save_dir_root, 'run', 'run_*')))
    run_id = int(runs[-1].split('_')[-1]) + 1 if runs else 0

save_dir = os.path.join(save_dir_root, 'run', 'run_' + str(run_id))
modelName = 'VNN_Fusion'
saveName = modelName + '-' + dataset


def train_model(dataset=dataset, save_dir=save_dir, num_classes=num_classes, lr=lr,
                num_epochs=nEpochs, save_epoch=snapshot, useTest=useTest, test_interval=nTestInterval):

    model_RGB = vnn_rgb_of_highQv2.VNN(num_classes=num_classes, num_ch=3, pretrained=False)
    model_OF = vnn_rgb_of_highQv2.VNN(num_classes=num_classes, num_ch=2, pretrained=False)
    model_fuse = vnn_fusion_highQv2.VNN_F(num_classes=num_classes, num_ch=192, pretrained=False)

    train_params = [
        {'params': vnn_rgb_of_highQv2.get_1x_lr_params(model_RGB), 'lr': lr},
        {'params': vnn_rgb_of_highQv2.get_1x_lr_params(model_OF), 'lr': lr},
        {'params': vnn_fusion_highQv2.get_1x_lr_params(model_fuse), 'lr': lr},
        {'params': vnn_fusion_highQv2.get_10x_lr_params(model_fuse), 'lr': lr},
    ]

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(train_params, lr=lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)

    if resume_epoch == 0:
        print("Training {} from scratch...".format(modelName))
    else:
        checkpoint = torch.load(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar'),
            map_location=lambda storage, loc: storage
        )
        print("Initializing weights from: {}...".format(
            os.path.join(save_dir, 'models', saveName + '_epoch-' + str(resume_epoch - 1) + '.pth.tar')))
        model_RGB.load_state_dict(checkpoint['state_dict_rgb'])
        model_OF.load_state_dict(checkpoint['state_dict_of'])
        model_fuse.load_state_dict(checkpoint['state_dict_fuse'])
        optimizer.load_state_dict(checkpoint['opt_dict'])

    print('Total params: %.2fM' % (
        (sum(p.numel() for p in model_RGB.parameters()) +
         sum(p.numel() for p in model_OF.parameters()) +
         sum(p.numel() for p in model_fuse.parameters())) / 1000000.0))

    model_RGB.to(device)
    model_OF.to(device)
    model_fuse.to(device)
    criterion.to(device)

    wandb.init(name=saveName, config={
        'dataset': dataset, 'lr': lr, 'num_epochs': num_epochs,
        'save_epoch': save_epoch, 'batch_size': 16,
    })

    print('Training model on {} dataset...'.format(dataset))
    train_dataloader = DataLoader(VideoDataset(dataset=dataset, split='train', clip_len=16, augment=True),
                                  batch_size=16, shuffle=True, num_workers=4)
    val_dataloader   = DataLoader(VideoDataset(dataset=dataset, split='val',   clip_len=16, augment=False),
                                  batch_size=16, num_workers=4)
    test_dataloader  = DataLoader(VideoDataset(dataset=dataset, split='test',  clip_len=16, augment=False),
                                  batch_size=16, num_workers=4)

    trainval_loaders = {'train': train_dataloader, 'val': val_dataloader}
    trainval_sizes = {x: len(trainval_loaders[x].dataset) for x in ['train', 'val']}
    test_size = len(test_dataloader.dataset)

    for epoch in range(resume_epoch, num_epochs):
        for phase in ['train', 'val']:
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            if phase == 'train':
                scheduler.step()
                model_RGB.train()
                model_OF.train()
                model_fuse.train()
            else:
                model_RGB.eval()
                model_OF.eval()
                model_fuse.eval()

            for inputs, labels in tqdm(trainval_loaders[phase]):
                inputs_of = compute_optical_flow(inputs, 112, 112)

                inputs = Variable(inputs, requires_grad=True).to(device)
                inputs_of = Variable(inputs_of, requires_grad=True).to(device)
                labels = Variable(labels).to(device)
                optimizer.zero_grad()

                if phase == 'train':
                    outputs_rgb = model_RGB(inputs)
                    outputs_of = model_OF(inputs_of)
                    outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))
                else:
                    with torch.no_grad():
                        outputs_rgb = model_RGB(inputs)
                        outputs_of = model_OF(inputs_of)
                        outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / trainval_sizes[phase]
            epoch_acc = running_corrects.double() / trainval_sizes[phase]

            if phase == 'train':
                wandb.log({'train/loss': epoch_loss, 'train/acc': epoch_acc, 'epoch': epoch})
            else:
                wandb.log({'val/loss': epoch_loss, 'val/acc': epoch_acc, 'epoch': epoch})

            print("[{}] Epoch: {}/{} Loss: {} Acc: {}".format(phase, epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

        if epoch % save_epoch == (save_epoch - 1):
            os.makedirs(os.path.join(save_dir, 'models'), exist_ok=True)
            torch.save({
                'epoch': epoch + 1,
                'state_dict_rgb': model_RGB.state_dict(),
                'state_dict_of': model_OF.state_dict(),
                'state_dict_fuse': model_fuse.state_dict(),
                'opt_dict': optimizer.state_dict(),
            }, os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar'))
            print("Save model at {}\n".format(os.path.join(save_dir, 'models', saveName + '_epoch-' + str(epoch) + '.pth.tar')))

        if useTest and epoch % test_interval == (test_interval - 1):
            model_RGB.eval()
            model_OF.eval()
            model_fuse.eval()
            start_time = timeit.default_timer()

            running_loss = 0.0
            running_corrects = 0.0

            for inputs, labels in tqdm(test_dataloader):
                inputs_of = compute_optical_flow(inputs, 112, 112)
                inputs = inputs.to(device)
                inputs_of = inputs_of.to(device)
                labels = labels.to(device)

                with torch.no_grad():
                    outputs_rgb = model_RGB(inputs)
                    outputs_of = model_OF(inputs_of)
                    outputs = model_fuse(torch.cat((outputs_rgb, outputs_of), 1))

                probs = nn.Softmax(dim=1)(outputs)
                preds = torch.max(probs, 1)[1]
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / test_size
            epoch_acc = running_corrects.double() / test_size

            wandb.log({'test/loss': epoch_loss, 'test/acc': epoch_acc, 'epoch': epoch})

            print("[test] Epoch: {}/{} Loss: {} Acc: {}".format(epoch+1, nEpochs, epoch_loss, epoch_acc))
            stop_time = timeit.default_timer()
            print("Execution time: " + str(stop_time - start_time) + "\n")

    wandb.finish()


if __name__ == "__main__":
    train_model()
