for dataset in 'OfficeHome'
do
    for target in 'Art' 'Clipart' 'RealWorld' 'Product'; do
        sbatch --job-name=$dataset-$target sbatch_GPA.sh $dataset $target
    done
done