conda env create -f environment.yml
conda activate boltzgen-fasta-epitope
python run_boltzgen_from_fasta_epitope.py \
  --fasta target.fasta \
  --epitope 25-40 \
  --outdir workbench_fasta_epitope \
  --peptide-length 12..18 \
  --num-designs 50 \
  --budget 10