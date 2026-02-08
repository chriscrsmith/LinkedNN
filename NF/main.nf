params.each { k, v -> 
    println "- $k = $v"
}
println "=============================\n"

workflow {

    totalsims = 51000

    //msprime
    chan = Channel.of( 1..totalsims )
    chunkSize = Math.ceil(totalsims / params.queueSize) as int
    chunks = chan.collate(chunkSize, remainder=true)
    sims = Simulate_Ne(chunks)
    chunks_only = sims.chunks_only
    allsims = chunks_only.collect()
    fanout = allsims.flatMap { it }  // emits (e.g., 51k) tuples from the single, collected list; extract first element from each
    chunks = fanout.collate(chunkSize, remainder=true)
    prep = Preprocess_msprime(chunks)

    // summary statistics
    chan = Channel.of( 1..totalsims )
    chunkSize = Math.ceil(totalsims / params.queueSize) as int
    chunks = chan.collate(chunkSize, remainder=true)
    sumstats = Calculate_SumStats(chunks)
    
}



process Simulate_Ne {

    // resources
    memory '4 GB'
    time '6h'

    // misc. settings
    //publishDir "${params.pubdir}/Ne/Targets/", pattern: "target_*.npy", mode: 'copy'  // skipping publish, as it only activates if the whole batch completes leaving some simids in the batch unfinished, don't want to rerun the whole batch if it doesn't finish; need individual outputs right away
    //publishDir "${params.pubdir}/Ne/TreeSeqs/", pattern: "output_*_recap.trees", mode: 'symlink'
    conda "${params.condadir}"
    tag { "sim_${chunk_list[0]}_to_${chunk_list[-1]}" }
    debug true

    input:
    val chunk_list

    output:
    val chunk_list, emit: chunks_only  // only emit chunk_list

    script:
    def ids = chunk_list.join(' ')  // removes "[" from the string-formatted list
    """
    mkdir -p ${params.pubdir}/Ne/Targets/
    mkdir -p ${params.pubdir}/Ne/TreeSeqs/
    for id in ${ids}
    do
        python ${params.repo}/Misc/sim_demog.py \$id 10,1000 1e2,1e4 1e2,1e4 ${params.workdir}/Ne/  # store progress on scratch
        ln -sf ${params.workdir}/Ne/TreeSeqs/output_\$id"_"recap.trees ${params.pubdir}/Ne/TreeSeqs/  # too large->link
        cp ${params.workdir}/Ne/Targets/target_\$id.npy ${params.pubdir}/Ne/Targets/
    done
    """
}



process Preprocess_msprime {

    // resources
    memory '4 GB'
    time '1h'

    // misc. settings
    conda "${params.condadir}"
    tag { "sim_${chunk_list[0]}_to_${chunk_list[-1]}" }
    debug true

    input:
    val chunk_list

    output:
    val(chunk_list)

    script:
    """
    python ${params.repo}/linkedNN.py --wd ${params.pubdir}/Ne/ \
                                                      --seed 1 \
                                                      --preprocess \
                                                      --num_snps 5000 \
                                                      --n 10 \
                                                      --w 50 \
                                                      --l 1e8 \
                                                      --hold_out 1000 \
                                                      --simid ${chunk_list[0]},${chunk_list[-1]} \
                                                      1> stdout 2> stderr  # otherwise stdout to main NF command (due to absolute paths)
    """
}



process Calculate_SumStats {

    // resources
    memory '4 GB'
    time '1h'

    // misc. settings
    conda "${params.condadir}"
    tag { "sim_${chunk_list[0]}_to_${chunk_list[-1]}" }
    debug true

    input:
    val chunk_list

    output:
    val chunk_list

    script:
    def ids = chunk_list.join(' ')
    """
    for id in ${ids}
    do
        python ${params.repo}/Misc/sumstats.py ${params.pubdir}/Ne/ \${id}  1> stdout 2> stderr  # redirect, otherwise prints stdout to main NF process (due to absolute paths)
    done
    """
}


