for dataset in 'OfficeHome'
do
    for target in 'Art' 'Clipart' 'RealWorld' 'Product'; do
        sbatch --job-name=$dataset-$target sbatch.sh $dataset $target
    done
done