for dataset in 'DomainNet'
do
    for target in 'clipart'; do
        sbatch --job-name=$dataset-$target sbatch.sh $dataset $target
    done
done