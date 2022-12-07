# phase1-incontext
python3 phase2.py -o results/comve_temp1.json -i ./data/ComVE/test_all.csv -t comve -p 1
python3 phase2.py -o results/comve_temp2.json -i ./data/ComVE/test_all.csv -t comve -p 2
python3 phase2.py -o results/comve_temp3.json -i ./data/ComVE/test_all.csv -t comve -p 3
python3 phase2.py -o results/comve_temp5.json -i ./data/ComVE/test_all.csv -t comve -p 5
python3 phase2.py -o results/comve_temp6.json -i ./data/ComVE/test_all.csv -t comve -p 6
python3 phase2.py -o results/esnli_temp1.json -i ./data/e-SNLI/test_all.csv -t esnli -p 1
python3 phase2.py -o results/esnli_temp2.json -i ./data/e-SNLI/test_all.csv -t esnli -p 2
python3 phase2.py -o results/esnli_temp3.json -i ./data/e-SNLI/test_all.csv -t esnli -p 3
python3 phase2.py -o results/esnli_temp4.json -i ./data/e-SNLI/test_all.csv -t esnli -p 4
python3 phase2.py -o results/esnli_temp5.json -i ./data/e-SNLI/test_all.csv -t esnli -p 5
python3 phase2.py -o results/esnli_temp6.json -i ./data/e-SNLI/test_all.csv -t esnli -p 6
# phase1-cgmh
python3 phase2.py -o results/comve_cgmh_all.json -i ./data/ComVE/test_cgmh.csv -t comve -m all -p 3
python3 phase2.py -o results/comve_cgmh_top1.json -i ./data/ComVE/test_cgmh.csv -t comve -m top1 -p 3
python3 phase2.py -o results/esnli_cgmh_all.json -i ./data/e-SNLI/test_cgmh.csv -t esnli -m all -p 3
python3 phase2.py -o results/esnli_cgmh_top1.json -i ./data/e-SNLI/test_cgmh.csv -t esnli -m top1 -p 3
