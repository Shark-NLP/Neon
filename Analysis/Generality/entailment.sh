# Entailment
python3 ../Phase2/phase2.py -o results/entail_orginal.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m top1 -p 0
python3 ../Phase2/phase2.py -o results/entail_top1_temp1.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m top1 -p 1
python3 ../Phase2/phase2.py -o results/entail_all_temp1.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m all -p 1
python3 ../Phase2/phase2.py -o results/entail_top1_temp2.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m top1 -p 2
python3 ../Phase2/phase2.py -o results/entail_all_temp2.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m all -p 2
python3 ../Phase2/phase2.py -o results/entail_top1_temp3.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m top1 -p 3
python3 ../Phase2/phase2.py -o results/entail_all_temp3.json -i ./data/e-SNLI/test_entail_all.csv -t entail -m all -p 3