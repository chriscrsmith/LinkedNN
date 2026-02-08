# data generator code for training

import sys
import numpy as np
import torch
import msprime
import tskit
from linkedNN.process_input import vcf2genos

def cropper(ts, W, sample_width, edge_width, alive_inds):
    "Cropping the map, returning individuals inside sampling window"
    cropped = []
    left_edge = np.random.uniform(
        low=edge_width, high=W - edge_width - sample_width
    )
    right_edge = left_edge + sample_width
    bottom_edge = np.random.uniform(
        low=edge_width, high=W - edge_width - sample_width
    )
    top_edge = bottom_edge + sample_width

    for i in alive_inds:
        ind = ts.individual(i)
        loc = ind.location[0:2]
        if (
            loc[0] > left_edge
            and loc[0] < right_edge
            and loc[1] > bottom_edge
            and loc[1] < top_edge
        ):
            cropped.append(i)

    return cropped

def sample_ind(ts, sampled_inds, W, i, j):
    bin_size = W / args.sample_grid
    output_ind = None
    for ind in sampled_inds:
        indiv = ts.individual(ind)
        loc = indiv.location[0:2]
        if (
            loc[0] > (i * bin_size)
            and loc[0] < ((i + 1) * bin_size)
            and loc[1] > (j * bin_size)
            and loc[1] < ((j + 1) * bin_size)
        ):
            output_ind = ind
            break
    if (
        output_ind is None
    ):  # if no individuals in the current square, choose a random ind
        output_ind = np.random.choice(sampled_inds, 1, replace=False)

    return output_ind

def unpolarize(snps, args):
    "Change 0,1 encoding to major/minor allele. Also filter no-biallelic"
    alleles = {}
    for i in range(args.n*2):
        a = snps[i]
        alleles[a] = alleles.get(a, 0) + 1
    if len(alleles) != 2:
        return False
    #
    major, minor = list(alleles.keys())  # random order
    if alleles[major] < alleles[minor]:
        major, minor = minor, major
    #
    new_genotypes = []
    for i in range(args.n*2):
        if snps[i] == major:
            new_genotype = 0
        else:
            new_genotype = 1
        #
        new_genotypes.append(new_genotype)

    return new_genotypes

def empirical_sample(ts, sampled_inds, args, dataglob):
    locs = np.array(dataglob.empirical_locs)
    np.random.shuffle(locs)
    indiv_dict = {}  # tracking which indivs have been picked up already
    for i in sampled_inds:
        indiv_dict[i] = 0
    keep_indivs = []
    for pt in range(args.n):  # for each sampling location
        dists = {}
        for i in indiv_dict:
            ind = ts.individual(i)
            loc = ind.location[0:2]
            d = ((loc[0] - locs[pt, 0]) ** 2
                 + (loc[1] - locs[pt, 1]) ** 2) ** (
                     1 / 2
                 )
            dists[d] = i  # see what I did there?
        nearest = dists[min(dists)]
        ind = ts.individual(nearest)
        loc = ind.location[0:2]
        keep_indivs.append(nearest)
        del indiv_dict[nearest]

    return keep_indivs

def mutate(ts, args):
    mu = float(args.mu)
    counter = 0
    while ts.num_sites < (
        args.num_snps * 2
    ):  # extra SNPs because a few are likely  non-biallelic           
        counter += 1
        ts = msprime.sim_mutations(
            ts,
            rate=mu,
            random_seed=args.seed+counter,
            model=msprime.SLiMMutationModel(type=0),
            keep=True,
        )
        if counter == 10:
            print("\n\nsorry, Dude. Didn't generate enough snps. \n\n")
            sys.stdout.flush()
            exit()
        mu *= 10
    return ts

def sample_ts(filepath, args, dataglob):
    "The meat: load in and fully process a tree sequence"

    # read input
    ts = tskit.load(filepath)
    np.random.seed(args.seed)

    # trim down to L base pairs
    if ts.sequence_length > args.l:
        ts = ts.delete_intervals([[args.l, ts.sequence_length]], simplify=True)
        ts = ts.rtrim()
    
    # recapitate
    alive_inds = []
    for i in ts.individuals():
        alive_inds.append(i.id)
    if args.recapitate == "True":
        Ne = len(alive_inds)
        if ts.num_populations > 1:
            ts = ts.simplify()  # gets rid of weird, extraneous populations
        demography = msprime.Demography.from_tree_sequence(ts)
        demography[0].initial_size = Ne
        ts = msprime.sim_ancestry(
            initial_state=ts,
            recombination_rate=args.rho,
            demography=demography,
            start_time=ts.metadata["SLiM"]["generation"],  # cycle?
            random_seed=seed,
        )

    # crop map
    if args.use_locs:
        sample_width = args.w
        sampled_inds = cropper(ts,
                               args.w,
                               sample_width,
                               0,
                               alive_inds)
        if len(sampled_inds) < args.n:
            print("\tnot enough samples, killed while-loop after 100 loops",
                  flush=True)
            exit()
    else:
        sampled_inds = list(alive_inds)

    # sample individuals
    if args.sample_grid is not None:
        if args.n < args.sample_grid**2:
            print("your sample grid is too fine, \
            not enough samples to fill it")
            exit()
        keep_indivs = []
        for r in range(
            int(np.ceil(args.n / args.sample_grid**2))
        ):  # sampling from each square multiple times until >= n samples
            for i in range(args.sample_grid):
                for j in range(args.sample_grid):
                    new_guy = sample_ind(ts, sampled_inds, args.w, i, j)
                    keep_indivs.append(new_guy)
                    sampled_inds.remove(new_guy)  # avoid sampling same guy
        keep_indivs = np.random.choice(
            keep_indivs, args.n, replace=False
        )  # taking n from the >=n list
    elif dataglob.empirical_locs is not None:
        keep_indivs = empirical_sample(
            ts, sampled_inds, args, dataglob
        )
    else:
        keep_indivs = np.random.choice(sampled_inds, args.n, replace=False)
    # 
    keep_nodes = []
    for i in keep_indivs:
        ind = ts.individual(i)
        keep_nodes.extend(ind.nodes)

    # simplify
    ts = ts.simplify(keep_nodes)

    # mutate
    if args.skip_mutate is False:
        ts = mutate(ts, args)

    # grab spatial locations
    if args.use_locs:    
        sample_dict = {}
        locs = []
        for samp in ts.samples():
            node = ts.node(samp)
            indID = node.individual
            if indID not in sample_dict:
                sample_dict[indID] = 0
                loc = ts.individual(indID).location[0:2]
                locs.append(loc)

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
        locs = locs.T
    else:
        locs = None

    # grab genos and positions
    geno_mat = ts.genotype_matrix()
    pos_mat = ts.sites_position

    # change 0,1 encoding to major/minor allele  
    if args.polarize is False:
        geno_mat_NEW = []
        pos_mat_NEW = []
        for s in range(ts.num_sites):
            new_genotypes = unpolarize(geno_mat[s,:], args)
            if new_genotypes is not False:
                geno_mat_NEW.append(new_genotypes)                             
                pos_mat_NEW.append(pos_mat[s])
        #
        geno_mat = np.array(geno_mat_NEW)
        pos_mat = np.array(pos_mat_NEW)
        if geno_mat.shape[0] < args.num_snps:
            raise ValueError("not enough SNPs after filtering")
                
    # sample SNPs
    mask = [True] * args.num_snps + \
        [False] * (geno_mat.shape[0] - args.num_snps)
    np.random.shuffle(mask)                                           
    geno_mat = geno_mat[mask, :]                                    
    pos_mat = pos_mat[mask]                                             
    
    # collapse genotypes, change to minor allele dosage (e.g. 0,1,2)
    if args.phase is False:
        geno_mat_NEW = np.zeros((args.num_snps, args.n))
        for i in range(args.n):
            geno_mat_NEW[:, i] += geno_mat[:, i*2]
            geno_mat_NEW[:, i] += geno_mat[:, i*2+1]
        #
        geno_mat = geno_mat_NEW.copy()
        
    # normalize positions by genome length
    pos_mat /= args.l
    
    # swap axes to get channels first
    geno_mat = geno_mat.T

    return geno_mat, pos_mat, locs


def data_generator(simids, sb, eb, args, dataglob):
    "Generates data containing batch_size samples"

    X1 = torch.empty((eb-sb, args.n, args.num_snps))  # genos
    X2 = torch.empty((eb-sb, 1, args.num_snps))  # pos
    Y = torch.empty((eb-sb, args.output_size), dtype=float)
    X3 = torch.empty((eb-sb, args.n, 2))  # locs
    
    ### augment by shuffling individuals within each preprocessed dataset
    individual_indices = np.arange(args.n)
    if args.skip_shuffle is False:
        np.random.shuffle(individual_indices)

    ### read data
    for i, ID in enumerate(simids[sb:eb]):
        
        # load target
        if args.empirical is None:
            Y[i,:] = torch.tensor(np.load(dataglob.targets[ID]))

        # load and re-order genos
        if args.empirical is None:
            genomat = np.load(dataglob.genos[ID])  # fp
            posmat = np.load(dataglob.pos[ID])
        else:
            genomat,posmat = vcf2genos(args) #args.empirical + ".vcf", args.n, args.num_snps, args.phase
        if dataglob.empirical_locs is None:
            genomat = genomat[individual_indices, :]
        X1[i, :, :] = torch.tensor(genomat)
        X2[i, 0, :] = torch.tensor(posmat)

        # load and re-order locs
        if args.use_locs:
            locs = np.load(dataglob.locs[ID]).T
            if dataglob.empirical_locs is None:
                locs = locs[individual_indices, :]
            X3[i, :, :] = torch.tensor(locs)

    return X1.float(), X2.float(), X3.float(), Y.float()  # float() to convert np float64 to torch float32
