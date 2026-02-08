# e.g., python sumstats.py output_dir/ <rep>

import numpy as np
import allel
import sys,os

### define inputs
fp = sys.argv[1]
idx = int(sys.argv[2])



def calcSS(geno_f):
    simid = geno_f.split("/")[-1].split(".")[0]
    fp_ss = fp_idx_ss + simid + ".ss.txt"
    if os.path.isfile(fp_ss):
        pass
    else:
    
        ### load genos
        arr = np.load(geno_f)
        arr = np.array(arr, dtype=int)
        arr = arr.T
        m,n = arr.shape
        geno = np.zeros((m,n,2), dtype=int)
        for snp in range(m):
            for indiv in range(n):
                if arr[snp, indiv] == 0:
                    pass
                elif arr[snp, indiv] == 1:
                    geno[snp, indiv] = [0,1]
                elif arr[snp, indiv] == 2:
                    geno[snp, indiv] = [1,1]
                else:
                    print("issue")
                    exit()

        ### preprocess genos
        geno = allel.GenotypeArray(geno)
        ac = geno.count_alleles()
        af = ac.to_frequencies()
        gn = geno.to_n_alt()

        ### calc stats
        pi = allel.mean_pairwise_difference(ac)
        pi_mean = np.nanmean(pi)
        pi_var = np.var(pi)
        tajima_D_mean = allel.tajima_d(ac)
        tajima_D_var = allel.moving_tajima_d(ac, size=100, step=100)
        tajima_D_var = np.nanvar(tajima_D_var)
        ho = allel.heterozygosity_observed(geno)
        ho_mean = np.nanmean(ho)
        ho_var = np.nanvar(ho)
        he = allel.heterozygosity_expected(af, ploidy=2)
        he_mean = np.nanmean(he)
        he_var = np.nanvar(he)
        F = allel.inbreeding_coefficient(geno)
        F_mean = np.nanmean(F)
        F_var = np.nanvar(F)
        r = allel.rogers_huff_r(gn)
        r2 = r**2
        r2_mean = np.nanmean(r2)  # in case monomorhphic snps causing nan
        r2_var = np.nanvar(r2)
        sfs = allel.sfs_folded(ac, n=n*2)  # (n only matters if missing data I think)
        sfs = sfs[1:]  # index 0 is monomorphic sites I think
        
        ### output
        outline = [pi_mean,
                   pi_var,
                   tajima_D_mean,
                   tajima_D_var,
                   ho_mean,
                   ho_var,
                   he_mean,
                   he_var,
                   F_mean,
                   F_var,
                   r2_mean,
                   r2_var]
        outline += list(sfs)
        outline = ",".join(map(str,outline))
        with open(fp_ss, "w") as outfile:
            outfile.write(outline + "\n")


            
### parse `Train/` dir
fp_idx_geno = fp + "/Train/Genos/"
fp_idx_ss = fp + "/Train/SumStats/"
os.makedirs(fp_idx_ss, exist_ok=True)
for root, subdir, files in os.walk(fp_idx_geno):
    if subdir == []:
        for geno_f in files:
            if geno_f == str(idx) + ".genos.npy":
                calcSS(root +"/"+ geno_f)

### parse `Test/` dir                
fp_idx_geno = fp + "/Test/Genos/"
fp_idx_ss = fp + "/Test/SumStats/"
os.makedirs(fp_idx_ss, exist_ok=True)
for root, subdir, files in os.walk(fp_idx_geno):
    if subdir == []:
        for geno_f in files:
            if geno_f == str(idx) + ".genos.npy":
                calcSS(root +"/"+ geno_f)


