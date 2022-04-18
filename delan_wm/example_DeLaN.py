import argparse
import torch
import numpy as np
import time
import easydict, yaml

from deep_lagrangian_networks.DeLaN_model import DeepLagrangianNetwork
from deep_lagrangian_networks.replay_memory import PyTorchReplayMemory
from deep_lagrangian_networks.utils import load_dataset, init_env
from dataset import data
from world_model import WorldModel

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
if __name__ == "__main__":

    args = easydict.EasyDict(yaml.load(open('./cfg.yaml'), yaml.FullLoader))
    seed, device, load_model, save_model = init_env(args)

    # Read the dataset:
    n_dof = 2
    train_data, test_data, divider, _ = load_dataset()

    # return (train_labels, train_qp, train_qv, train_qa, train_p, train_pd, train_tau), \
    #      (test_labels, test_qp, test_qv, test_qa, test_p, test_pd, test_tau, test_m, test_c, test_g),\
    #      divider, dt_mean
    train_labels, train_qp, train_qv, train_qa, _, __, train_tau = train_data  ####
    test_labels, test_qp, test_qv, test_qa, _, __, test_tau, test_m, test_c, test_g = test_data  ###

    '''
    # Load existing model parameters:
    if load_model:
        load_file = "data/delan_model.torch"
        state = torch.load(load_file)

        delan_model = DeepLagrangianNetwork(n_dof, **state['args'])
        delan_model.load_state_dict(state['state_dict'])
    else:
        # Construct DeLaN:
        delan_model = DeepLagrangianNetwork(n_dof, **args)
    delan_model = delan_model.to(device) #if cuda else delan_model.cpu()
    '''
    delan_model = WorldModel(args).to(device)

    # Generate & Initialize the Optimizer:
    optimizer = torch.optim.Adam(delan_model.parameters(),
                                 lr=args["learning_rate"],
                                 weight_decay=args["weight_decay"],
                                 amsgrad=True)

    # Generate Replay Memory:
    # mem_dim = ((n_dof, ), (n_dof, ), (n_dof, ), (n_dof, ))
    # mem = PyTorchReplayMemory(train_qp.shape[0], args["n_minibatch"], mem_dim, device)
    # mem.add_samples([train_qp, train_qv, train_qa, train_tau])

    train_data = data(args.train_data_path)
    test_data = data(args.test_data_path)
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=args.n_minibatch, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=args.n_minibatch, shuffle=True, num_workers=4)
    # Start Training Loop:
    t0_start = time.perf_counter()

    epoch_i = 0
    while epoch_i < args['max_epoch']:
        l_mem_mean_inv_dyn, l_mem_var_inv_dyn = 0.0, 0.0
        l_mem_mean_dEdt, l_mem_var_dEdt = 0.0, 0.0
        l_mem, n_batches = 0.0, 0.0

        # for q,v,next_q,next_v,tau in train_data:
        for rew_state, q, v, next_q, next_v, contact_force,action_force, next_tau in train_loader:
            # q,v,next_q,next_v,tau=batch
            delan_model.train()
            q, v, next_q, next_v, contact_force,action_force, next_tau, rew_state = q.to(device), v.to(device), next_q.to(device), \
                                                             next_v.to(device), contact_force.to(device), action_force.to(device),next_tau.to(
                device), rew_state.to(device)

            t0_batch = time.perf_counter()

            optimizer.zero_grad()

            # Compute the Rigid Body Dynamics Model:
            # q_pred, v_pred, dEdt_hat, tau_pred = delan_model(q, v, tau, rew_state)
            q_pred, v_pred, dEdt_hat, tau_pred = delan_model(q, v, contact_force, action_force, rew_state)

            # Compute the loss of the Euler-Lagrange Differential Equation:
            # err_inv = torch.sum((tau_hat - tau) ** 2, dim=1)
            err_inv = torch.sum((next_q - q_pred) ** 2, dim=1) + torch.sum((next_v - v_pred) ** 2, dim=1)
            l_mean_inv_dyn = torch.mean(err_inv)
            l_var_inv_dyn = torch.var(err_inv)

            # Compute the loss of the Power Conservation:
            # dEdt: qd * tau, cannot obtained currently, replaced by MSE of tau_pred
            # print(v.shape,tau.shape)
            # dEdt = torch.matmul(v.view(-1, args.dof, 1).transpose(dim0=1, dim1=2), tau.view(-1, args.dof, 1)).view(-1)
            # err_dEdt = (dEdt_hat - dEdt) ** 2

            err_dEdt = (tau_pred - next_tau[:, :-args.dof]) ** 2
            l_mean_dEdt = torch.mean(err_dEdt)
            l_var_dEdt = torch.var(err_dEdt)

            # loss = l_mean_inv_dyn + l_mem_mean_dEdt #original code, may have bug?
            loss = l_mean_inv_dyn + l_mean_dEdt

            loss.backward()
            optimizer.step()

            # Update internal data:
            n_batches += 1
            if n_batches % 100 == 0:
                print("-------{} loss {:.4f}, Inv loss {:.4f}, Energy loss {:.4f}".format(n_batches, loss.item(),
                                                                                          l_mean_inv_dyn.item(),
                                                                                          l_mean_dEdt.item()))
                # print("-------------running means:{:.4f},{:.4f},{:.4f}",delan_model.q_bn.running_mean,delan_model.qd_bn.running_mean,delan_model.tau_bn.running_mean)
                # print("-------------running vars:{:.4f},{:.4f},{:.4f}",delan_model.q_bn.running_var,delan_model.qd_bn.running_var,delan_model.tau_bn.running_var)
            l_mem += loss.item()
            l_mem_mean_inv_dyn += l_mean_inv_dyn.item()
            l_mem_var_inv_dyn += l_var_inv_dyn.item()
            l_mem_mean_dEdt += l_mean_dEdt.item()
            l_mem_var_dEdt += l_var_dEdt.item()  # tensorboard

            t_batch = time.perf_counter() - t0_batch

        # test
        if epoch_i % 2 == 0:
            for rew_state, q, v, next_q, next_v, contact_force, action_force, next_tau in test_loader:
                delan_model.eval()
                loss_test, batch_cnt, loss_test_inv, loss_test_dedt = 0, 0, 0, 0

                q, v, next_q, next_v, contact_force, action_force, next_tau, rew_state = q.to(device), v.to(
                    device), next_q.to(device), next_v.to(device), contact_force.to(device), action_force.to(
                    device), next_tau.to(device), rew_state.to(device)

                q_pred, v_pred, dEdt_hat, tau_pred = delan_model(q, v, contact_force,action_force, rew_state)

                err_inv = torch.sum((next_q - q_pred) ** 2, dim=1) + torch.sum((next_v - v_pred) ** 2, dim=1)
                l_mean_inv_dyn = torch.mean(err_inv)
                l_var_inv_dyn = torch.var(err_inv)

                err_dEdt = (tau_pred - next_tau[:, :-args.dof]) ** 2
                l_mean_dEdt = torch.mean(err_dEdt)
                l_var_dEdt = torch.var(err_dEdt)

                loss = l_mean_inv_dyn + l_mean_dEdt
                loss_test += loss
                loss_test_inv += l_mean_inv_dyn
                loss_test_dedt += l_mean_dEdt
                batch_cnt += 1
            print('<->Test: Loss {:.4f}, Inv Loss {:.4f}, Energy loss {:.4f}'.format(loss_test, loss_test_inv,
                                                                                         loss_test_dedt))
            print('<->Sample pos: real ', next_q[0], 'pred: ',q_pred[0],'mse:',torch.sum((next_q[0]-q_pred[0])**2).item())
            print('<->Sample vel: real ', next_v[0], 'pred: ',v_pred[0],'mse:',torch.sum((next_v[0]-v_pred[0])**2).item())
        # Update Epoch Loss & Computation Time:
        l_mem_mean_inv_dyn /= float(n_batches)
        l_mem_var_inv_dyn /= float(n_batches)
        l_mem_mean_dEdt /= float(n_batches)
        l_mem_var_dEdt /= float(n_batches)
        l_mem /= float(n_batches)
        epoch_i += 1

        if epoch_i == 1 or np.mod(epoch_i, 2) == 0:
            print("Epoch {0:05d}: ".format(epoch_i), end="")
            # print("Time={0:05.1f}s".format(time.perf_counter() - t0_start), end=", ")
            print("Loss={0:.3e}".format(l_mem), end=", ")
            print("Inv Dyn={0:.3e} \u00B1{1:.3e}".format(l_mem_mean_inv_dyn, 1.96 * np.sqrt(l_mem_var_inv_dyn)),
                  end=", ")
            print("Power Con={0:.3e} \u00B1{1:.3e}".format(l_mem_mean_dEdt, 1.96 * np.sqrt(l_mem_var_dEdt)))
            # print('test:',tau_pred[0,:])

        # if save_model is not None:
        if save_model:
            torch.save({"epoch": epoch_i,
                        "args": args,
                        "state_dict": delan_model.state_dict()},
                        "delan_model.torch")
