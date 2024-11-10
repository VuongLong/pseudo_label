for dataset in 'ImageCLEF'
do
    for target in 'c' 'i' 'p'; do
        sbatch --job-name=$dataset-$target sbatch_clef.sh $dataset $target
    done
done