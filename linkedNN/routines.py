### nn architecture, plus train and test loops

from linkedNN.data_generation import data_generator
import matplotlib.pyplot as plt
import os
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import sys
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from linkedNN.models import ld_layer
from linkedNN.models import pairwise_cnn
from linkedNN.models import cnn

    
def training_loop(args, dataglob):

    # split into val,train sets                                            
    sim_ids = np.arange(0, len(dataglob.targets))
    train, val = train_test_split(sim_ids, test_size=args.validation_split)
    partition = {}
    partition["train"] = list(train)
    partition["val"] = list(val)
    
    # initialize model
    train_path = args.wd + "/Train/model_" + str(args.seed) + ".sav"
    if os.path.isfile(train_path):
        if args.force is True:
            print("training output exists; overwriting...")
        else:
            print("training output exists. Can overwrite with --force")
            sys.exit(1)
    if args.module == "pairwise_cnn":
        print("using pairwise conv")
        model = pairwise_cnn(args)
    elif args.module == "cnn":
        print("using basic cnn")
        model = cnn(args)
    elif args.module == "ld_layer":
        print("using ld_layer")
        model = ld_layer(args.n, args.output_size, args.l)
    else:
        print("unrecognized module name")
        sys.exit(1)        
    print(model, flush=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)
    model.to(device)

    # more training params
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=args.patience, factor=0.5)
    train_n = len(partition["train"])
    val_n = len(partition["val"])
    mse = nn.MSELoss()
    loss_val_best = 999999
    best_epoch = 9999
    train_losses = []
    val_losses = []    
    epochs_wo_improvement = 0
    optimizer.zero_grad()
    if args.grad_acc is None:
        args.grad_acc = int(args.batch_size)
    
    for epoch in range(args.max_epochs):
        train_loss=0.0
        model.train()
        shuffled_inds=np.array(partition["train"])[torch.randperm(train_n)]

        # loop over mini batches
        for b in range(int(np.ceil(float(train_n)/float(args.batch_size)))):
            #print("\tbatch", b, "out of", int(np.ceil(train_n/args.batch_size)), flush=True)
            sb = b*args.batch_size
            eb = min((b+1)*args.batch_size, train_n)
                
            # loop through even smaller batches for gradient accumulation
            accum_loss = 0.0
            accum_steps = int(np.ceil(float(eb - sb) / float(args.grad_acc)))
            for a in range(accum_steps):
                sa = sb + a * args.grad_acc
                ea = sb + min((a+1)*args.grad_acc, eb-sb)
                #print("\t\tgrad accumulation", a, "out of", accum_steps, ";", sa, "to", ea, flush=True)
                X_input_1, X_input_2, _, Y_true = data_generator(shuffled_inds, sa, ea, args, dataglob)
                X_input_1 = X_input_1.to(device)
                X_input_2 = X_input_2.to(device)
                Y_true = Y_true.to(device)

                # forward
                Y_est = model(X_input_1, X_input_2)
                if epoch==0 and b==0 and a==0:  # print total params (after forward pass due to lazy linear) 
                    total_params = sum(p.numel() for p in model.parameters())
                    print(f"Total parameters: {total_params}")
                loss = mse(Y_est, Y_true)
                accum_loss += loss.item() * (ea-sa)
                loss *= float(ea-sa) / float(eb-sb)  # scale loss by accumulation size  

                # backprop
                loss.backward()
                
            #
            optimizer.step()
            optimizer.zero_grad()
            train_loss += accum_loss  # for logging
                
        # finalize training loss
        train_loss /= train_n

        # validation
        val_loss=0.0
        model.eval()
        with torch.no_grad():
            for b in range(int(np.ceil(float(val_n)/float(args.batch_size)))):
                sb = b*args.batch_size
                eb = min((b+1)*args.batch_size, val_n)
                
                # loop through even smaller batches for gradient accumulation                                 
                accum_loss = 0.0
                accum_steps = int(np.ceil(float(eb - sb) / float(args.grad_acc)))
                for a in range(accum_steps):
                    sa = sb + a * args.grad_acc
                    ea = sb + min((a+1)*args.grad_acc, eb-sb)
                    X_input_1, X_input_2, _, Y_true = data_generator(partition["val"], sb, eb, args, dataglob)
                    X_input_1 = X_input_1.to(device)
                    X_input_2 = X_input_2.to(device)
                    Y_true = Y_true.to(device)
                    Y_est = model(X_input_1, X_input_2)
                    loss = mse(Y_est, Y_true)
                    accum_loss += loss.item() * (ea-sa)
                    loss *= float(ea-sa) / float(eb-sb)  # scale loss by accumulation size

                #
                val_loss += accum_loss
                val_losses.append(val_loss)

            # examine val loss
            val_loss /= val_n
            scheduler.step(val_loss)
            if val_loss < loss_val_best:
                epochs_wo_improvement = 0
                loss_val_best=np.array(val_loss)
                best_epoch = epoch
                torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'loss': train_loss,
                        'loss_val': val_loss,
                        }, train_path)

            else:
                epochs_wo_improvement += 1
                if epochs_wo_improvement == args.early_stop:
                    print("Early stopping on epoch", epoch, ". No improvement for", str(args.early_stop), "epochs.", flush=True)
                    break                    

            # print updates
            print("finished training epoch", epoch+1,
                  "lr: ", optimizer.param_groups[0]["lr"],
                  "train_loss: ", train_loss,
                  "val_loss:", val_loss,
                  "best_val_loss:", loss_val_best,
                  flush=True)
            torch.save({'train_losses': train_losses,
                        'val_losses': val_losses,
                        'model_state_dict_fs': model.state_dict(),
                        }, train_path+'fs')
            
    print("final train_loss:",train_loss,"val_loss:",val_loss,"val loss best:",loss_val_best, flush=True)
    
    return


def test_loop(args, dataglob):

    if args.empirical is None:
        outfile = args.wd + "/Test/predictions_" + str(args.seed) + ".npy"
    else:
        outfile = args.wd + "/Test/empirical_" + str(args.seed) + ".npy"        
    if os.path.isfile(outfile):
        if args.force is True:
            print("pred output exists; overwriting...")
            os.remove(outfile)
        else:
            print("test output exists. Can overwrite with --force")
            sys.exit(1)
    
    train_path = args.wd + "/Train/model_" + str(args.seed) + ".sav"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device, flush=True)
    checkpoint=torch.load(train_path, weights_only=False, map_location=device)
    if args.module == "pairwise_cnn":
        print("using pairwise conv")
        model = pairwise_cnn(args)
    elif args.module == "cnn":
        print("using regular cnn")
        model = cnn(args)
    elif args.module == "ld_layer":
        print("using ld_layer")
        model = ld_layer(args.n, args.output_size, args.l)
    else:
        print("unrecognized module name")
        sys.exit(1)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    print("using saved model from epoch", epoch, flush=True)
    model.to(device)
    mse = nn.MSELoss()
    if args.grad_acc is None:
        args.grad_acc = int(args.batch_size)

    # loop over batches
    if args.empirical is None:
        test_n = len(dataglob.targets)
    else:
        test_n = args.num_reps
    simids = np.arange(test_n)
    data = []
    model.eval()
    with torch.no_grad():
        for b in range(int(np.ceil(float(test_n)/float(args.batch_size)))):
            sb = b*args.batch_size
            eb = min((b+1)*args.batch_size, test_n)

            # loop through even smaller batches for gradient accumulation
            accum_steps = int(np.ceil(float(eb - sb) / float(args.grad_acc)))
            for a in range(accum_steps):                
                sa = sb + a * args.grad_acc
                ea = sb + min((a+1)*args.grad_acc, eb-sb)
                print("\ttest indices", sa, "to", ea-1, "out of", test_n)
                X_input_1, X_input_2, _, Y_true = data_generator(simids, sb, eb, args, dataglob)
                X_input_1 = X_input_1.to(device)
                X_input_2 = X_input_2.to(device)
                Y_est = model(X_input_1, X_input_2)
                Y_est = Y_est.float().cpu().numpy()
                prediction = (Y_est * dataglob.sdTarg) + dataglob.meanTarg   
                prediction = np.exp(prediction)
                if args.empirical is None:
                    Y_true = Y_true.float().cpu().numpy()
                    trueval = (Y_true * dataglob.sdTarg) + dataglob.meanTarg  # un-normalize (broadcasting over batch)
                    trueval = np.exp(trueval)  # undo log
                    new_data = np.stack([trueval, prediction, Y_true, Y_est], axis=0)  # (4, bsz, output_size)
                else:
                    new_data = np.stack(prediction, axis=0)  # (bsz, output_size)
                #
                data.append(new_data)

    # write
    if args.empirical is None:
        data = np.concatenate(data, axis=1)
    else:
        data = np.concatenate(data, axis=0)
        m = np.mean(data, axis=0)
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)
        for targ in range(data.shape[-1]):
            print("target-"+str(targ) + ":",
                  m[targ],
                  "(" + str(mins[targ]) + "," + str(maxs[targ]) + ")",
                  )
    #
    np.save(outfile, data)

    # plot
    if args.empirical is None:
        dec = 3  # decimal places in error metrics
        data = torch.tensor(data)
        ave_rmse,ave_r2,ave_mrae = [],[],[]
        for o in range(args.output_size):
            # rmse = np.sqrt(mse(data[0,:,o], data[1,:,o]))
            # ave_rmse.append(rmse)
            # rmse =str(np.round(np.array(rmse), dec))
            # print("output", o, "RMSE (no-logged):", rmse)
            # r2 = r2_score(np.log(data[0,:,o]), np.log(data[1,:,o]))
            # ave_r2.append(r2)
            # r2 =str(np.round(np.array(r2), dec))
            # print("output", o, "r2 (log):", r2)
            mrae = torch.mean(torch.abs((data[0,:,o] - data[1,:,o]) / data[0,:,o]))
            ave_mrae.append(mrae)
            mrae =str(np.round(np.array(mrae), dec))
            print("target", o, "MRAE (no-logged):", mrae)
            plt.scatter(np.log10(data[0,:,o].flatten()), np.log10(data[1,:,o].flatten()))
            fs = 16
            #plt.xlabel(r"True $N_e$", fontsize=fs)
            #plt.ylabel(r"Estimated $N_e$", fontsize=fs)
            #plt.title("Testing on held-out simulations", fontsize=fs)
            plt.xlabel("True value, target #" + str(o), fontsize=fs)
            plt.ylabel("Estimated value, target #" + str(o), fontsize=fs)
            # line
            x_min, x_max = plt.xlim()
            y_min, y_max = plt.ylim()
            line_start = max(x_min, y_min)
            line_end = min(x_max, y_max)
            line_coords = np.array([line_start, line_end])
            plt.plot(line_coords, line_coords, color='gray', linestyle='--')
            #
            plt.text(.01, .99,
                     "MRAE="+mrae, #'output-'+str(o)+ '\n' + 'RMSE='+rmse + "\n" + "r2="+r2  + "\n" + "MRAE="+mrae,
                     ha='left', va='top',
                     fontsize=fs,
                     transform=plt.gca().transAxes,) # (0,1 range))
            tick_locations = [np.log10(1e2), np.log10(1e3), np.log10(1e4)]
            tick_labels = [r"$10^2$", r"$10^3$", r"$10^4$"]
            plt.xticks(tick_locations, tick_labels)
            plt.yticks(tick_locations, tick_labels)
            plt.tight_layout()                                                                                                                            
            plt.savefig(args.wd + "/Test/results_" + str(args.seed) + "_output"+str(o) + ".pdf",
                        format="pdf", bbox_inches="tight")
            plt.close()

    return
