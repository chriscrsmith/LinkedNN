# helper utils for reading in data

import numpy as np
import os
from dataclasses import dataclass


# reads a list of filepaths, stores in dict
def read_dict(path):
    collection, counter = {}, 0
    with open(path) as infile:
        for line in infile:
            newline = line.strip()
            collection[counter] = newline
            counter += 1
    return collection


# reads a list of floats, stores in list
def read_single_value(path):
    collection = []
    with open(path) as infile:
        for line in infile:
            with open(line.strip()) as individual_file:
                newline = float(individual_file.readline().strip())
                collection.append(newline)
    return collection


# reads a list of floats, stores in dict
def read_single_value_dict(path):
    collection, counter = {}, 0
    with open(path) as infile:
        for line in infile:
            with open(line.strip()) as individual_file:
                newline = float(individual_file.readline().strip())
                collection[counter] = newline
                counter += 1
    return collection


# numpy-load list of floats, store as list
def load_single_value(path):
    collection = []
    with open(path) as infile:
        for line in infile:
            newline = np.load(line.strip())
            collection.append(newline)
    return collection


# numpy-load list of floats, store as dict
def load_single_value_dict(path):
    collection, counter = {}, 0
    with open(path) as infile:
        for line in infile:
            newline = np.load(line.strip())
            collection[counter] = newline
            counter += 1
    return collection


# read table of lat+long coords, store in list
def read_locs(path):
    collection = []
    with open(path) as infile:
        for line in infile:
            newline = line.strip().split()
            newline = list(map(float, newline))
            collection.append(newline)
    return collection


# fill a dictionary with a single value
def fill_dict_single_value(val, reps):
    collection = {}
    for i in range(reps):
        collection[i] = val
    return collection


# convert list to dictionary with ordinal keys
def dict_from_list(mylist):
    collection = {}
    for i in range(len(mylist)):
        collection[i] = mylist[i]
    return collection


# convert list to keys in a dictionary, values all zero
def list2dict(mylist):
    collection = {}
    for i in range(len(mylist)):
        collection[mylist[i]] = 0
    return collection


# read input paths from a preprocessed, hierarchical folder
def dict_from_preprocessed(args):
    targets, genos, pos, locs, counter = {}, {}, {}, {}, 0
    if args.train is True:
        for root, subdir, files in os.walk(args.wd + "/Train/Genos/"):
            if subdir == []:
                for f in files:
                    genopath = os.path.join(root, f)
                    targetpath = "Targets".join(genopath.rsplit("Genos",1)) 
                    targetpath = "target".join(targetpath.rsplit("genos",1))
                    locpath = "Locs".join(genopath.rsplit("Genos",1))     
                    locpath = "locs".join(locpath.rsplit("genos",1))
                    pospath = "Positions".join(genopath.rsplit("Genos",1))
                    pospath = "pos".join(pospath.rsplit("genos",1))
                    targets[counter] = targetpath
                    genos[counter] = genopath
                    locs[counter] = locpath
                    pos[counter] = pospath
                    counter += 1
    elif args.predict is True:
        for root, subdir, files in os.walk(args.wd+"/Test/Genos/"):
            if subdir == []: 
                for f in files:
                    genopath = os.path.join(root, f)
                    targetpath = genopath.replace("Genos", "Targets").replace("genos","target")
                    locpath = genopath.replace("Genos", "Locs").replace("genos","locs")
                    pospath = genopath.replace("Genos", "Positions").replace("genos","pos")
                    targets[counter] = targetpath
                    genos[counter] = genopath
                    locs[counter] = locpath
                    pos[counter] = pospath
                    counter += 1

    return targets, genos, pos, locs


# class for organizing data passed b/t fxns
@dataclass                           
class DataBundle:               
    targets: np.ndarray = None       
    trees: list = None               
    genos: np.ndarray = None         
    pos: np.ndarray = None           
    locs: np.ndarray = None          
    empirical_locs: np.ndarray = None
    meanTarg: float = None
    sdTarg: float = None

    
def catalog_sims(args):
    
    # check out how many sims and targets
    targets = {}
    sims = {}
    if os.path.isdir(args.wd + "/Targets/") is False:
        print("can't find Targets/")
        exit()
    if os.path.isdir(args.wd + "/TreeSeqs/") is False:
        print("can't find Treeseqs/")
        exit()
    for root, subdir, files in os.walk(args.wd+"/TreeSeqs/"):
        if subdir == []:
            for f in files:
                if "_recap.trees" in f:
                    sim_path = os.path.join(root, f)
                    simid = int(sim_path.split("/")[-1].split("_")[1])
                    sims[simid] = sim_path
                    #
                    target_path = sim_path.replace("TreeSeqs","Targets").replace("output","target").replace("_recap.trees", ".npy")
                    targid = int(target_path.split("/")[-1].split("_")[-1].split(".")[0])
                    targets[targid] = target_path

    # maybe save fps somewhere for safe keeping?

    return sims, targets

    
