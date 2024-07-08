import  os
os.environ["CUDA_VISIBLE_DEVICES"]="4"

import time
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from HyperTools import *
from Model.SACNet import *
from RCCA import *

DataName = {1: 'PaviaU', 2: 'Salinas', 3: 'IndinaP', 4: 'HoustonU', 5: 'xqh'}

def set_dataset_parameters(args):
    """Set parameters based on the dataset ID."""
    if args.dataID == 1:
        return 9, 103, 610, 340, './Data/PaviaU/'
    elif args.dataID == 2:
        return 16, 204, 512, 217, './Data/Salinas/'
    elif args.dataID == 3:
        return 16, 200, 145, 145, './Data/IndianP/'
    elif args.dataID == 4:
        return 15, 144, 349, 1905, './Data/HoustonU/'
    elif args.dataID == 5:
        return 6, 310, 456, 352, './Data/xqh/'
    else:
        raise ValueError("Invalid dataID")

def load_data(dataID, train_samples):
    """Load HSI data based on the dataset ID."""
    X, Y, train_array, test_array = LoadHSI(dataID, train_samples)
    Y -= 1
    return X, Y, train_array, test_array

def initialize_model(args, num_features, num_classes, m, n):
    """Initialize the model based on the given arguments."""
    if args.model == 'SACNet':
        return SACNet(num_features=num_features, num_classes=num_classes)
    elif args.model == 'RCCA':
        prior_size = [int((((m - 12) / 2) - 12) / 2), int((((n - 12) / 2) - 12) / 2)]
        return RCCA(num_features=num_features, prior_size=prior_size, num_classes=num_classes), prior_size
    else:
        raise ValueError("Invalid model")

def train_model(model, images, label, optimizer, criterion, num_epochs, args):
    """Train the model with the given data and parameters."""
    for epoch in range(num_epochs):
        adjust_learning_rate(optimizer, args.lr, epoch, num_epochs)
        optimizer.zero_grad()
        output, context_prior_map = model(images)
        seg_loss = criterion(output, context_prior_map, label)
        seg_loss.backward()
        optimizer.step()
        if (epoch + 1) % 1 == 0:
            print(f'epoch {epoch + 1}/{num_epochs}: cls_loss = {seg_loss.item():.3f}')

def evaluate_model(model, images, test_array, Y, num_classes, save_path_prefix, model_name, args, mode='clean'):
    """Evaluate the model and save the results."""
    model.eval()
    output, context_prior_map = model(images)
    _, predict_labels = torch.max(output, 1)
    predict_labels = np.squeeze(predict_labels.detach().cpu().numpy()).reshape(-1)
    OA, AA, kappa, ProducerA = CalAccuracy(predict_labels[test_array], Y[test_array])
    img = DrawResult(np.reshape(predict_labels + 1, -1), args.dataID)
    plt.imsave(f'{save_path_prefix}{model_name}_{mode}_OA{int(OA * 10000)}_kappa{int(kappa * 10000)}.png', img)
    print(f'OA={OA * 100:.3f}, AA={AA * 100:.3f}, Kappa={kappa * 100:.3f}')
    print(f'producerA: {ProducerA}')
    return OA, AA, kappa, ProducerA

def perform_attack(model, images, args, Y_tar, num_classes, prior_size, epsilon, num_features, h, w):
    """Perform adversarial attack and return adversarial examples."""
    if args.attack == 'FGSM':
        processed_image = Variable(images).requires_grad_()
        label = torch.zeros((1, h, w)).long().cuda()  # Create a zero target with correct dimensions
        criterion = myLoss(num_classes=num_classes, down_sample_size=prior_size).cuda()
        output, context_prior_map = model(processed_image)
        seg_loss = criterion(output, context_prior_map, label)
        seg_loss.backward()
        adv_noise = epsilon * processed_image.grad.data / torch.norm(processed_image.grad.data, float("inf"))
        processed_image.data = processed_image.data - adv_noise
        X_adv = torch.clamp(processed_image, 0, 1).cpu().data.numpy()[0]
        X_adv = np.reshape(X_adv, (1, num_features, h, w))

        return torch.from_numpy(X_adv).float().cuda()

def main(args):
    num_classes, num_features, m, n, save_pre_dir = set_dataset_parameters(args)
    iter = args.iter
    OA_clean, OA_attack = np.zeros(iter), np.zeros(iter)
    AA_clean, AA_attack = np.zeros(iter), np.zeros(iter)
    Kappa_clean, Kappa_attack = np.zeros(iter), np.zeros(iter)
    CA_clean, CA_attack = np.zeros((num_classes, iter)), np.zeros((num_classes, iter))

    for eep in range(iter):
        X, Y, train_array, test_array = load_data(args.dataID, args.train_samples)
        _, h, w = X.shape
        X_train = np.reshape(X, (1, num_features, h, w))
        Y_train = np.ones(Y.shape) * 255
        Y_train[train_array] = Y[train_array]
        Y_train = np.reshape(Y_train, (1, h, w))

        Y_tar = np.zeros(Y.shape)
        Y_tar = np.reshape(Y_tar, (1, h, w))

        save_path_prefix = args.save_path_prefix + 'Exp_' + DataName[args.dataID] + '/'
        os.makedirs(save_path_prefix, exist_ok=True)
        model, prior_size = initialize_model(args, num_features, num_classes, m, n)

        model = model.cuda()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.decay)
        images = torch.from_numpy(X_train).float().cuda()
        label = torch.from_numpy(Y_train).long().cuda()
        criterion = myLoss(num_classes=num_classes, down_sample_size=prior_size).cuda()

        # Train model
        train_model(model, images, label, optimizer, criterion, args.num_epochs, args)

        # Evaluate on clean data
        OA, AA, kappa, ProducerA = evaluate_model(model, images, test_array, Y, num_classes, save_path_prefix, args.model, args, 'clean')
        OA_clean[eep], AA_clean[eep], Kappa_clean[eep] = OA, AA, kappa
        CA_clean[0:num_classes, eep] = ProducerA

        # Perform adversarial attack
        adv_images = perform_attack(model, images, args, Y_tar, num_classes, prior_size, args.epsilon, num_features, h, w)

        # Evaluate on adversarial data
        OA, AA, kappa, ProducerA = evaluate_model(model, adv_images, test_array, Y, num_classes, save_path_prefix, args.model, args, 'FGSM')
        OA_attack[eep], AA_attack[eep], Kappa_attack[eep] = OA, AA, kappa
        CA_attack[0:num_classes, eep] = ProducerA

    # Log and print final results
    log_results(OA_clean, AA_clean, Kappa_clean, CA_clean, OA_attack, AA_attack, Kappa_attack, CA_attack)

def log_results(OA_clean, AA_clean, Kappa_clean, CA_clean, OA_attack, AA_attack, Kappa_attack, CA_attack):
    """Log and print the final results."""
    print('===============Clean===============')
    print(f'OA={np.average(OA_clean) * 100:.3f}, AA={np.average(AA_clean) * 100:.3f}, Kappa={np.average(Kappa_clean) * 100:.3f}')
    print(f'OA_std={np.std(OA_clean) * 100:.3f}, AA_std={np.std(AA_clean) * 100:.3f}, Kappa_std={np.std(Kappa_clean) * 100:.3f}')
    print(f'producerA: {np.average(CA_clean, 1)}')
    print(f'producerA_std: {np.std(CA_clean, 1)}')
    print('===============Attack===============')
    print(f'OA={np.average(OA_attack) * 100:.3f}, AA={np.average(AA_attack) * 100:.3f}, Kappa={np.average(Kappa_attack) * 100:.3f}')
    print(f'OA_std={np.std(OA_attack) * 100:.3f}, AA_std={np.std(AA_attack) * 100:.3f}, Kappa_std={np.std(Kappa_attack) * 100:.3f}')
    print(f'producerA: {np.average(CA_attack, 1)}')
    print(f'producerA_std: {np.std(CA_attack, 1)}')

if __name__ == '__main__':
    # DataName = {1:'PaviaU',2:'Salinas',3:'IndinaP',4:'HoustonU',5:'xqh';6:KSC}

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataID', type=int, default=3)
    parser.add_argument('--save_path_prefix', type=str, default='./')
    parser.add_argument('--model', type=str, default='RCCA')

    parser.add_argument('--attack', type=str, default='FGSM') # 1.FGSM, 2.C&W
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--decay', type=float, default=5e-5)
    parser.add_argument('--epsilon', type=float, default=0.04)
    parser.add_argument('--train_samples', type=int, default=100)
    parser.add_argument('--iter', type=int, default=10)
    parser.add_argument('--num_epochs', type=int, default=500)

    main(parser.parse_args())
