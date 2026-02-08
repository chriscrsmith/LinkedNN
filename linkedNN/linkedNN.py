#!/usr/bin/env python

# main code for linkedNN

import argparse
import os
import sys
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from linkedNN.data_generation import sample_ts
from linkedNN.check_args import check_args
from linkedNN.read_input import read_locs
from linkedNN.read_input import dict_from_preprocessed
from linkedNN.read_input import DataBundle
from linkedNN.read_input import catalog_sims
from linkedNN.process_input import project_locs
from linkedNN.routines import training_loop
from linkedNN.routines import test_loop


parser = argparse.ArgumentParser()
parser.add_argument(
    "--preprocess",
    action="store_true",
    default=False,
    help="create preprocessed tensors from tree sequences",
)
parser.add_argument(
    "--train", action="store_true", default=False, help="run training pipeline"
)
parser.add_argument(
    "--predict",
    action="store_true",
    default=False,
    help="run prediction pipeline"
)
parser.add_argument(
    "--empirical",
    default=None,
    help="prefix for vcf and locs"
)
parser.add_argument("--simid", default=None, type=str,
                    help="specific simulation id for preprocessing: 1-indexed, \
                    corresponds to line number in tree_list.txt")
parser.add_argument(
    "--edge_width",
    help="crop a fixed width from each edge of the map; \
    enter 'sigma' to set edge_width equal to sigma",
    default="0",
    type=str,
)
parser.add_argument(
    "--num_snps",
    default=None,
    type=int,
    help="number of SNPs",
)
parser.add_argument(
    "--n",
    default=None,
    type=int,
    help="sample size",
)
parser.add_argument(
    "--mu",
    help="beginning mutation rate: mu is increased until num_snps is achieved",
    default=1e-12,
    type=float,
)
parser.add_argument(
    "--l",
    help="length of genome in bp",
    default=None,
    type=float,
)
parser.add_argument(
    "--rho",
    help="recombination rate",
    default=1e-8,
    type=float)
parser.add_argument(
    "--w",
    default=None,
    type=int,
    help="map width (will try to parse automatically if \"W\" provided to SLiM)",
)
parser.add_argument(
    "--num_samples",
    default=1,
    type=int,
    help="number of repeated samples (each of size n) from each tree sequence",
)
parser.add_argument(
    "--num_reps",
    default=1,
    type=int,
    help="number of replicate-draws from the empirical genotype matrix of each sample",
)
parser.add_argument(
    "--hold_out",
    default=None,
    type=int,
    help="integer, the number of tree sequences to hold out for testing.",
)
parser.add_argument(
    "--validation_split",
    default=0.2,
    type=float,
    help="proportion of training set (after holding out test data) \
    to use for validation during training.",
)
parser.add_argument(
    "--batch_size",
    default=10,
    type=int,
    help="batch size for training")
parser.add_argument(
    "--grad_acc",
    default=None,
    type=int,
    help="subset-batch size for gradient accumulation: should be smaller than batch size")
parser.add_argument(
    "--max_epochs", default=1000, type=int, help="max epochs for training"
)
parser.add_argument(
    "--patience",
    type=int,
    default=10,
    help="n epochs to run the optimizer after last improvement in \
    validation loss before cutting LR in half.",
)
parser.add_argument(
    "--early_stop",
    type=int,
    default=50,
    help="n epochs to run the optimizer after last improvement in \
    validation loss before stopping training.",
)
parser.add_argument(
    "--dropout",
    default=0,
    type=float,
    help="proportion of weights to zero at the dropout layer.",
)
parser.add_argument(
    "--recapitate",
    action="store_true",
    help="recapitate tree sequences",
    default=False,
)
parser.add_argument(
    "--skip_mutate",
    action="store_true",
    help="skip adding mutations",
    default=False,
)
parser.add_argument(
    "--wd",
    help="file name stem for working directory with all inputs and outputs",
    default=None,
    type=str
)
parser.add_argument("--seed", default=None, type=int, help="random seed.")
parser.add_argument(
    "--plot_history",
    default=False,
    type=str,
    help="plot training history? default: False",
)
parser.add_argument(
    "--load_weights",
    default=None,
    type=str,
    help="Path to a .weights.h5 file to load weight from previous run.",
)
parser.add_argument(
    "--phase",
    default=False,
    action="store_true",
    help="phase genotype, default is unknown phase"
)
parser.add_argument(
    "--polarize",
    default=False,
    help="polarize the snps; default is no polarization",
    action="store_true",
)
parser.add_argument(
    "--keras_verbose",
    default=1,
    type=int,
    help="verbose argument passed to keras in model training. \
    0 = silent. 1 = progress bars for minibatches. \
    2 = show epochs. \
    Yes, 1 is more verbose than 2. Blame keras.",
)
parser.add_argument(
    "--threads",
    default=1,
    type=int,
    help="num threads.",
)
parser.add_argument(
    "--learning_rate",
    default=1e-4,
    type=float,
    help="learning rate.",
)
parser.add_argument("--grid_coarseness", help="TO DO", default=50, type=int)
parser.add_argument(
    "--sample_grid",
    help="coarseness of grid for grid-sampling",
    default=None,
    type=int
)
parser.add_argument(
    "--upsample",
    help="number of upsample layers",
    default=6,
    type=int)
parser.add_argument(
    "--pairs",
    help="number of pairs to include in the feature block",
    type=int,
    default=None,
)
parser.add_argument(
    "--pairs_encode",
    help="number of pairs (<= pairs_encode) to use for gradient \
    in the first part of the network",
    type=int,
    default=None,
)
parser.add_argument(
    "--ld_prop_encode",
    help="proportion of SNP pairs to use for gradient with ld_pairwise model",
    type=float,
    default=1.0,
)
parser.add_argument(
    "--force",
    action="store_true",
    default=False,
    help="force overwrite of existing output"
)
parser.add_argument(
    "--skip_shuffle",
    action="store_true",
    help="skip shuffling individuals within each dataset to get different pairs each batch and augment the training set (default is to shuffle)",
    default=False,
)
parser.add_argument(
    "--snp_clusters",
    help="simulate random non-uniform SNP clusters",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--fixed_rateMap",
    help="simulate a fixed, predefined mutation map",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--module",
    help="name of nn model",
    type=str,
    default="ld_layer",
)
parser.add_argument(
    "--skip_positions",
    help="avoid using SNP positions",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--use_locs",
    help="use sampling locations",
    action="store_true",
    default=False,
)
args = parser.parse_args()
check_args(args)
if len(sys.argv) == 1:
    parser.print_help()
    sys.exit(1)


def preprocess():
    # read inputs
    print("cataloging inputs...")
    trees,target_paths = catalog_sims(args)
    total_sims = len(trees)

    # separate training and test data
    os.makedirs(args.wd, exist_ok=True)
    tt_path = args.wd + "/train_test_split.npy"
    if not os.path.isfile(tt_path):
        print("saving test indices in working dir", flush=True)
        if 0 in trees:  # 0-indexing
            train, test = train_test_split(np.arange(total_sims),
                                           test_size=args.hold_out)
        else:  # 1-indexing
            print("1")
            train, test = train_test_split(np.arange(total_sims)+1,
                                           test_size=args.hold_out)
        np.save(tt_path, np.array([train, test], dtype=object)) 
    else:
        print("loading existing test indices", flush=True)
        arr = np.load(tt_path, allow_pickle=True)
        train, test = arr[0],arr[1]    
        
    # loop through training targets to get mean and sd
    if os.path.isfile(args.wd + "/preprocess_params.npy"):
        print("loading saved mean and sd from before", flush=True)
        arr = np.load(
            args.wd + "/preprocess_params.npy",
            allow_pickle=True,)
        n, num_snps, meanTarg, sdTarg, train, test, l, output_size = list(arr)
    else:
        print("calculating mean and sd on the training set saved in", tt_path, flush=True)
        targets = []
        counter = 1
        for i in train:
            print("looking at target number", counter, flush=True)
            counter+=1
            arr = np.load(target_paths[i])
            arr = np.log(arr)
            output_size = len(arr)  # the number of targets/outputs the NN will predict 
            targets.append(arr)
        #
        targets = np.array(targets)
        meanTarg = np.mean(targets, axis=0)
        sdTarg = np.std(targets, axis=0)
        np.save(args.wd + "/preprocess_params",
                np.array([args.n, args.num_snps, meanTarg, sdTarg, train, test, args.l, output_size], dtype=object)
                )

    # make directories
    os.makedirs(os.path.join(args.wd,
                             "Train/Targets",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Train/Genos",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Train/Positions",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Train/Locs",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Test/Targets",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Test/Genos",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Test/Positions",
                             str(args.seed)),
                exist_ok=True)
    os.makedirs(os.path.join(args.wd,
                             "Test/Locs",
                             str(args.seed)),
                exist_ok=True)

    # process
    if args.simid is None:
        preprocess_list = trees
    else:
        if "," in args.simid:
            start,end = list(map(int,args.simid.split(",")))
            preprocess_list = list(range(start, end+1))
        else:
            preprocess_list = [int(args.simid)]
    for i in preprocess_list:
        print("preprocessing simid", i, flush=True)
        if i in test:
            split = "Test"
        else:
            split = "Train"
        targetfile = os.path.join(
            args.wd, split, "Targets", str(args.seed), str(i) + ".target"
        )
        genofile = os.path.join(
            args.wd, split, "Genos", str(args.seed), str(i) + ".genos"
        )
        locfile = os.path.join(
            args.wd, split, "Locs", str(args.seed), str(i) + ".locs"
        )
        posfile = os.path.join(
            args.wd, split, "Positions", str(args.seed), str(i) + ".pos"
        )
        if (
            os.path.isfile(genofile + ".npy") is False
            or os.path.isfile(locfile + ".npy") is False
            or os.path.isfile(posfile + ".npy") is False
            or os.path.isfile(targetfile + ".npy") is False
        ):
            if args.empirical is not None:
                locs = read_locs(args.empirical + ".locs")
                if len(locs) != args.n:
                    print("length of locs file doesn't match n", flush=True)
                    exit()
                locs = project_locs(locs, trees[i])
            else:
                locs = None
            dataglob = DataBundle(empirical_locs=locs)
            geno_mat, pos, locs = sample_ts(trees[i], args, dataglob)
            np.save(genofile, geno_mat)
            np.save(posfile, pos)
            np.save(locfile, locs)
        if (
            os.path.isfile(genofile + ".npy") is True
            and os.path.isfile(posfile + ".npy") is True
            and os.path.isfile(locfile + ".npy") is True
        ):  # (only add target if inputs successful)
            if os.path.isfile(targetfile + ".npy") is False:
                target = np.load(target_paths[i])
                target = np.log(target)
                target = (target - meanTarg) / sdTarg
                np.save(targetfile, target)

    return


def train():

    # read targets
    print("reading input paths", flush=True)
    targets, genos, pos, locs = dict_from_preprocessed(args)

    # grab train/test split and other params from preprocessed dir
    print("grab mean and sd", flush=True)
    arr = np.load(
        args.wd + "/preprocess_params.npy",
        allow_pickle=True,)
    n, num_snps, meanTarg, sdTarg, train, test, l, output_size = list(arr)
    args.n, args.num_snps, args.l, args.output_size = int(n), int(num_snps), int(l), int(output_size)

    # save training params:
    print("save params", flush=True)
    if args.pairs is None:
        args.pairs = int((args.n * (args.n - 1)) / 2)
    if args.pairs_encode is None:
        args.pairs_encode = int(args.pairs)
    np.save(
        args.wd + "/Train/training_params_" + str(args.seed),
        np.array([args.seed,
                  args.max_epochs,
                  args.validation_split,
                  args.learning_rate,
                  args.pairs, args.pairs_encode,
                  args.n, args.num_snps,
                  meanTarg, sdTarg,
                  train, test,
                  args.batch_size,
                  args.module,
                  args.skip_positions,
                  args.l,
                  args.output_size,
                  ], object)
    )
    
    # organize inputs for generator
    dataglob = DataBundle(targets=targets,
                          genos=genos,
                          pos=pos,
                          locs=locs,
                          )
    
    # train
    training_loop(args, dataglob)
    
    return


def predict():

    # grab mean and sd from training distribution
    arr = np.load(
        args.wd + "/preprocess_params.npy",
        allow_pickle=True,)
    n, num_snps, meanTarg, sdTarg, train, test, l, output_size = list(arr)
    args.n, args.num_snps, args.l, args.output_size = int(n), int(num_snps), int(l), int(output_size)

    # grab saved training params
    params = np.load(args.wd + "/Train/training_params_"
                     + str(args.seed) + ".npy", allow_pickle=True)
    args.pairs, args.pairs_encode = int(params[4]), int(params[5])
    
    # load inputs
    targets, genos, pos, locs = dict_from_preprocessed(args)
    
    # organize inputs for generator
    dataglob = DataBundle(targets=targets,
                          genos=genos,
                          pos=pos,
                          locs=locs,
                          meanTarg=meanTarg,
                          sdTarg=sdTarg,
                          )
    
    # predict
    print("predicting", flush=True)
    test_loop(args, dataglob)
    
    return


def empirical():
    # grab mean and sd from training distribution
    arr = np.load(
        args.wd + "/preprocess_params.npy",
        allow_pickle=True,)
    n, num_snps, meanTarg, sdTarg, train, test, l, output_size = list(arr)
    args.n, args.num_snps, args.l, args.output_size = int(n), int(num_snps), int(l), int(output_size)

    # grab saved training params                                  
    params = np.load(args.wd + "/Train/training_params_"
                     + str(args.seed) + ".npy", allow_pickle=True)
    args.pairs, args.pairs_encode = int(params[4]), int(params[5])
    
    # project locs
    if args.use_locs is True: 
        locs = read_locs(args.empirical + ".locs")
        locs = np.array(locs)
        if len(locs) != args.n:
            print("length of locs file doesn't match n", flush=True)
            exit()
        locs = project_locs(locs)

        # rescale locs
        locs = np.array(locs)
        minx = min(locs[:, 0])
        maxx = max(locs[:, 0])
        miny = min(locs[:, 1])
        maxy = max(locs[:, 1])
        x_range = maxx - minx
        y_range = maxy - miny
        locs[:, 0] = (locs[:, 0] - minx) / x_range  # rescale to (0,1)
        locs[:, 1] = (locs[:, 1] - miny) / y_range
        if x_range > y_range:  # these four lines for preserving aspect ratio
            locs[:, 1] *= y_range / x_range
        elif x_range < y_range:
            locs[:, 0] *= x_range / y_range

    # organize inputs for generator         
    dataglob = DataBundle(targets=None,
                          genos=None,
                          pos=None,
                          locs=None,
                          meanTarg=meanTarg,
                          sdTarg=sdTarg,
                          )

    # predict                               
    print("predicting", flush=True)
    test_loop(args, dataglob)
            
    return


def plot_history():
    loss, val_loss = [], [
        np.nan
    ]  # loss starts at iteration 0; val_loss starts at end of first epoch
    with open(args.plot_history) as infile:
        for line in infile:
            if "val_loss:" in line:
                endofline = line.strip().split(" loss:")[-1]
                loss.append(float(endofline.split()[0]))
                val_loss.append(float(endofline.split()[3]))
    loss.append(np.nan)  # make columns even-length
    epochs = np.arange(len(loss))
    fig = plt.figure(figsize=(4, 1.5), dpi=200)
    plt.rcParams.update({"font.size": 7})
    ax1 = fig.add_axes([0, 0, 0.4, 1])
    ax1.plot(epochs, val_loss, color="blue", lw=0.5, label="val_loss")
    ax1.set_xlabel("Epoch")
    ax1.plot(epochs, loss, color="red", lw=0.5, label="loss")
    ax1.legend()
    fig.savefig(args.plot_history + "_plot.pdf", bbox_inches="tight")


def run():

    # main #
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # pre-process
    if args.preprocess is True:
        print("starting pre-processing pipeline", flush=True)
        preprocess()

    # train
    if args.train is True:
        print("starting training pipeline", flush=True)
        train()

    # plot training history
    if args.plot_history:
        plot_history()

    # predict
    if args.predict is True:
        print("starting prediction pipeline", flush=True)
        if args.empirical is None:
            print("predicting on simulated data", flush=True)
            predict()
        else:
            print("predicting on empirical data", flush=True)
            empirical()
