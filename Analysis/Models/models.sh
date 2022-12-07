python3 models.py -l gpt2-xl -o results/gpt2-xl_gt_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l ../gpt2-large -o results/gpt2-large_gt_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l ../gpt2-medium -o results/gpt2-medium_gt_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l gpt2-xl -o results/gpt2-xl_top1_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "top1" -ml 30
python3 models.py -l ../gpt2-large -o results/gpt2-large_top1_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "top1" -ml 30
python3 models.py -l ../gpt2-medium -o results/gpt2-medium_top1_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "top1"
python3 models.py -l gpt2-xl -o results/gpt2-xl_all_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "all" -ml 30
python3 models.py -l ../gpt2-large -o results/gpt2-large_all_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "all" -ml 30
python3 models.py -l ../gpt2-medium -o results/gpt2-medium_all_3.json -i 3 -f ./data/ComVE/test_all.csv --mode "all" -ml 30
python3 models.py -l gpt2-xl -o results/gpt2-xl_baseline.json -i 0 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l ../gpt2-large -o results/gpt2-large_baseline.json -i 0 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l ../gpt2-medium -o results/gpt2-medium_baseline.json -i 0 -f ./data/ComVE/test_all.csv --mode "gt" -ml 30
python3 models.py -l gpt2-xl -o esnli/gpt2-xl_gt_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../gpt2-large -o esnli/gpt2-large_gt_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../gpt2-medium -o esnli/gpt2-medium_gt_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l gpt2-xl -o esnli/gpt2-xl_top1_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "top1" -ml 30 -t esnli
python3 models.py -l ../gpt2-large -o esnli/gpt2-large_top1_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "top1" -ml 30 -t esnli
python3 models.py -l ../gpt2-medium -o esnli/gpt2-medium_top1_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "top1" -t esnli
python3 models.py -l gpt2-xl -o esnli/gpt2-xl_all_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "all" -ml 30 -t esnli
python3 models.py -l ../gpt2-large -o esnli/gpt2-large_all_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "all" -ml 30 -t esnli
python3 models.py -l ../gpt2-medium -o esnli/gpt2-medium_all_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "all" -ml 30 -t esnli
python3 models.py -l gpt2-xl -o esnli/gpt2-xl_baseline.json -i 0 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../gpt2-large -o esnli/gpt2-large_baseline.json -i 0 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../gpt2-medium -o esnli/gpt2-medium_baseline.json -i 0 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../models/opt/opt-13b -o esnli/opt-13b_gt_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli
python3 models.py -l ../models/opt/opt-13b -o esnli/opt-13b_top1_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "top1" -ml 30 -t esnli
python3 models.py -l ../models/opt/opt-13b -o esnli/opt-13b_all_3.json -i 3 -f ./data/e-SNLI/test_all.csv --mode "all" -ml 30 -t esnli
python3 models.py -l ../models/opt/opt-13b -o esnli/opt-13b_baseline.json -i 0 -f ./data/e-SNLI/test_all.csv --mode "gt" -ml 30 -t esnli