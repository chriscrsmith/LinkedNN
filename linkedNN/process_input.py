# helper functions for processing inputs

import numpy as np
import sys
import random
import tskit
import utm

# project sample locations
def project_locs(locs, fp=None):
    # projection (plus some code for calculating error)
    locs = np.array(locs)
    locs = np.array(utm.from_latlon(locs[:, 0], locs[:, 1])[0:2]) / 1000
    locs = locs.T

    # calculate extremes
    min_lat = min(locs[:, 0])
    min_long = min(locs[:, 1])
    max_lat = max(locs[:, 0])
    max_long = max(locs[:, 1])
    lat_range = max_lat - min_lat
    long_range = max_long - min_long

    # rescale lat and long to each start at 0
    locs[:, 0] = (
        1 - ((locs[:, 0] - min_lat) / lat_range)
    ) * lat_range  # "1-" to orient north-south
    locs[:, 1] = locs[:, 1] - min_long

    # reposition sample locations to random area within the map
    if fp:
        ts = tskit.load(fp)
        left_edge = np.random.uniform(low=0, high=args.w - long_range)
        bottom_edge = np.random.uniform(low=0, high=args.w - lat_range)
        locs[:, 0] += bottom_edge
        locs[:, 1] += left_edge

    return locs


# preprocess vcf
def vcf2genos(args):
    geno_mat = []
    pos_mat = []
    vcf = open(args.empirical + ".vcf", "r")
    for line in vcf:
        if line[0:2] == "##":
            pass
        elif line[0] == "#":
            header = line.strip().split("\t")
        else:
            newline = line.strip().split("\t")
            genos = []
            alleles = {}
            for field in range(9, len(newline)):
                pos = float(newline[1])
                pos /= args.l  # rescale to (0,1) using L from training
                geno = newline[field].split(":")[0].split("/")
                if "." in geno:  # missing
                    alleles.setdefault(-1, 0)
                    alleles[-1] += 1
                else:
                    alleles.setdefault(int(geno[0]), 0)
                    alleles[int(geno[0])] += 1
                    alleles.setdefault(int(geno[1]), 0)
                    alleles[int(geno[1])] += 1
                    genos.append( [int(geno[0]), int(geno[1])] )  # still phased at this point
            #
            if -1 not in alleles and len(list(alleles.keys())) == 2:  # filters non-missing AND biallelic
                
                # recalculate major/minor allele
                major, minor = list(alleles.keys())  # (random init)
                if alleles[major] < alleles[minor]:
                    major, minor = minor, major
                #                                                  
                new_genotypes = []
                for i in range(args.n):
                    new_genotype = [None, None]
                    for j in range(2):  # diploid
                        if genos[i][j] == major:
                            new_genotype[j] = 0
                        else:
                            new_genotype[j] = 1
                    #                                              
                    new_genotypes.append(new_genotype)
                #

                # collapse
                if args.phase is False:
                    for i in range(args.n):
                        new_genotypes[i] = sum(new_genotypes[i])

                #
                geno_mat.append(new_genotypes)
                pos_mat.append(pos)

    # check if enough snps
    if len(geno_mat) < args.num_snps:
        print("not enough snps", len(geno_mat))
        exit()

    # sample snps
    geno_mat = np.array(geno_mat)
    pos_mat = np.array(pos_mat)
    idx = np.random.choice(geno_mat.shape[0],
                            args.num_snps,
                            replace=False)
    geno_mat = geno_mat[idx]
    pos_mat = pos_mat[idx]
    geno_mat = np.swapaxes(geno_mat, 0, 1)
    
    return geno_mat,pos_mat




# main
def main():
    vcf_path = sys.argv[1]
    n = sys.argv[2]
    if n == "None":
        n = None
    else:
        n = int(n)
    num_snps = int(sys.argv[3])
    outname = sys.argv[4]
    phase = int(sys.argv[5])
    geno_mat = vcf2genos(vcf_path, n, num_snps, phase)
    np.save(outname + ".genos", geno_mat)


if __name__ == "__main__":
    main()
