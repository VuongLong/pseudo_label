for dataset in 'DomainNet'
do
    for target in 'clipart' 'infograph' 'painting' 'real' 'quickdraw' 'sketch'; do
        sbatch --job-name=$dataset-$target sbatch.sh $dataset $target
    done
done