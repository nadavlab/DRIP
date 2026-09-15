sbatch \
    --job-name=eval_PRScs \
    --mem=64G \
    --time=1-00:00:00 \
    --cpus-per-task=4 \
    --output=eval_PRScs_%j.out \
    --error=eval_PRScs_%j.err \
    --wrap="python /path/to/your/project//code/from_the_begining_in_bgu/simulations/PRS-CS/evaluating_PRSCS.py"