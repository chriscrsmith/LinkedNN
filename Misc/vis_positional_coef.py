# e.g., (linkedNN) chriscs@x1001c2s5b0n0:/N/u/chriscs/Software/LinkedNN> python Misc/vis_positional_coef.py 

import sys
sys.path.append("/N/u/chriscs/Software/LinkedNN/")
from routines import *
from dataclasses import dataclass
from data_generation import *
import numpy as np
from scipy.spatial.distance import pdist, squareform
import tskit
import msprime
import allel
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

##########################################
### visualize mapping fxn coefficients
##########################################

### params
#fp = "/N/slate/chriscs/Linkage/Ne_v6/Train/model_1653.sav"
fp = "/N/slate/chriscs/Linkage/Ne_v5/Train/model_1603.sav"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device, flush=True)
@dataclass
class args:
    skip_positions: bool = False
    targets: np.ndarray = None
    trees: list = None
    genos: np.ndarray = None
    pos: np.ndarray = None
    locs: np.ndarray = None
    empirical_locs: np.ndarray = None
    meanSig: float = None
    sdSig: float = None
    l: int = 1e8
    n: int = 10
    num_snps: int = 5000
    output_size: int = 3
    #output_size: int = 1
    mu: float = 1e-13
    seed: int = 123
    recapitate: bool = False
    use_locs: bool = False
    sample_grid: int = None
    skip_mutate: bool = False
    polarize: bool = False
    phase: bool = False
    snp_clusters:bool = False
    fixed_rateMap:bool = False


### load model
checkpoint=torch.load(fp, weights_only=False, map_location=device)
model = linkednn(args)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)
model.eval()

### distance grid
num_dists = 1000
positions_0 = torch.zeros((num_dists))
positions_1 = np.logspace( np.log10(1/args.l), np.log10(1), num=num_dists)  # log-U(1bp to 1e8)   this fxn uses log10 base and raises to exponent x and y
positions_1 = torch.from_numpy(positions_1).float()

x = torch.stack([positions_0,positions_1], axis=1)  # (num_dists, 2)
x = x.unsqueeze(0)  # (bsz, num_dists, 2)

### get coefficients
with torch.no_grad():
    coeffs = model.positional_mapping(x)  # (1, 64, 100)
    coeffs = coeffs.squeeze(0)
    F = coeffs.shape[0]

### count peaks between 5e5 and 5e6
print(coeffs.shape)
peaks = torch.max(coeffs, dim=-1)  # (index 0) peaks for each coefficient and (index 1) associated indices along the distance array
peak_idx = peaks[1]  # index 1 is the position along the distance array (not the max value)
dist = positions_1 - positions_0  # log-uniform
dist *= args.l
peak_dist = dist[peak_idx]
print(peak_dist)
start = 5e5
end = 5e6
print((peak_dist > start) & (peak_dist < end))
print( torch.sum((peak_dist > start) & (peak_dist < end)) )

### count peaks between 5e5 and 5e6                                                                                                
peaks = torch.max(coeffs, dim=-1)  # (index 0) peaks for each coefficient and (index 1) associated indices along the distance array
peak_idx = peaks[1]  # index 1 is the position along the distance array (not the max value)                                        
dist = positions_1 - positions_0  # log-uniform                                                                                    
dist *= args.l
peak_dist = dist[peak_idx]
start = 5
end = 50
print( torch.sum((peak_dist > start) & (peak_dist < end)) )

### count flat zero value coefficients
print( torch.sum( (peaks[0] == 0)) )

### preprocess dists
dist = torch.log10(x[0,:,1] - x[0,:,0])

### plot
for i in range(F):
    plt.plot(dist.numpy(), coeffs[i,:].detach().numpy(), alpha=0.2, lw=1, color="blue")
plt.xlabel("Log10 genomic distance (bp)")
plt.ylabel("Scaling coefficient", color="Blue")
plt.title("Learned coefficients for different genomic distances")

tick_locations = np.log10(np.logspace( np.log10(1/args.l), np.log10(1), num=9))
tick_labels = [r"$1$", r"$10$", r"$10^2$", r"$10^3$", r"$10^4$", r"$10^5$", r"$10^6$", r"$10^7$", r"$10^8$"]
plt.xticks(tick_locations, tick_labels)

######################
### vis LD decay
######################

simids = [0,1,2,3,4,5,6,7,8,9]  # fixed-Ne sims: small Ne, then big
colors = ["black"]*5 + ["grey"]*5  # small, then big

lines,labels = [],[]
ax2 = plt.gca().twinx()
ax2.set_ylabel(r"LD ($r^2$)")

for simid in simids:
    print("\tsimid", simid)

    ### process simulation into genotypes
    pref = "/N/slate/chriscs/Linkage/Fixed_NE_linkedNN_Fig1/"
    os.makedirs(pref, exist_ok=True)
    geno_fp = pref + "/" + str(simid) + ".genos.npy"
    pos_fp = pref + "/" + str(simid) + ".pos.npy"
    if os.path.exists(geno_fp) and os.path.exists(pos_fp):
        print("exists")
        pass
    else:
        print("muts")
        fp = pref + "/TreeSeqs/output_" + str(simid) + "_recap.trees"
        genos, pos, _ = sample_ts(fp, args, args)
        np.save(geno_fp, genos)
        np.save(pos_fp, pos)
    
    ### load genos for smaller Ne sim
    geno_fp = pref + "/" + str(simid) + ".genos.npy"
    arr = np.load(geno_fp).astype('int')
    arr = arr.T
    m,n = arr.shape
    genos = np.zeros((m,n,2), dtype=int)
    for snp in range(m):
        for indiv in range(n):
            if arr[snp, indiv] == 0:
                pass
            elif arr[snp, indiv] == 1:
                genos[snp, indiv] = [0,1]
            elif arr[snp, indiv] == 2:
                genos[snp, indiv] = [1,1]
            else:
                print("issue")
                exit()
    #
    g = allel.GenotypeArray(genos)
    gn = g.to_n_alt(fill=-1)

    ### load pos
    pos_fp = pref + "/" + str(simid) + ".pos.npy"
    pos = np.load(pos_fp)

    ### pw dists
    pos = np.expand_dims(pos,-1)
    dist = pdist(pos)

    ### calc LD
    r = allel.rogers_huff_r(gn[:, :])
    r2 = r**2

    ### discretize
    numbins=20
    min_val = 1e3/args.l
    max_val = 1e8/args.l
    bins = np.logspace(np.log10(min_val), np.log10(max_val), numbins)
    r2_means = []
    for i in range(len(bins)-1):
        mask = (dist>=bins[i]) & (dist < bins[i+1])
        print(bins[i], bins[i+1], np.sum(mask))
        r2_means.append(np.nanmean(r2[mask]))
        
    ### add to plot
    #line_0, = ax2.plot(np.log10(bins[1:]), r2_means, color=colors[simid], linestyle='--', label=f'Sim {simid}')
    midpt =  bins[:-1] + (bins[1:] - bins[:-1])/2
    line_0, = ax2.plot(np.log10(midpt), r2_means, color=colors[simid], linestyle='--', label=f'Sim {simid}')


custom_lines = [
    Line2D([0], [0], color='blue', linestyle='-', label='Learned coefficients'),
    Line2D([0], [0], color='black', linestyle='--', label=r'LD bins with $N_e=10^2$'),
    Line2D([0], [0], color='grey', linestyle='--', label=r'LD bins with $N_e=10^4$')
]
plt.legend(handles=custom_lines, loc="upper left")
    

### final plot params
#plt.legend(lines, labels, loc='upper left')
plt.tight_layout()                                                                    
plt.savefig("positional_coefficients.pdf")                                            
plt.close()                                                                               
