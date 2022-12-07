# ComVE task
python3 phase1.py --instance_file ../data/ComVE/train.csv --test_file sampled_instance_comve.csv --task comve --output_file phase1_comve_sampled_0.csv --max_tokens 25
python3 phase1.py --instance_file ../data/ComVE/train.csv --test_file sampled_instance_comve.csv --task comve --output_file phase1_comve_sampled_1.csv --max_tokens 25
python3 phase1.py --instance_file ../data/ComVE/train.csv --test_file sampled_instance_comve.csv --task comve --output_file phase1_comve_sampled_2.csv --max_tokens 25
python3 phase1.py --instance_file ../data/ComVE/train.csv --test_file sampled_instance_comve.csv --task comve --output_file phase1_comve_sampled_3.csv --max_tokens 25
# e-SNLI task
python3 phase1.py --instance_file ../data/e-snli/train.csv --test_file sampled_instance_esnli.csv --task esnli --output_file phase1_esnli_sampled_0.csv --max_tokens 40
python3 phase1.py --instance_file ../data/e-snli/train.csv --test_file sampled_instance_esnli.csv --task esnli --output_file phase1_esnli_sampled_1.csv --max_tokens 40
python3 phase1.py --instance_file ../data/e-snli/train.csv --test_file sampled_instance_esnli.csv --task esnli --output_file phase1_esnli_sampled_2.csv --max_tokens 40
python3 phase1.py --instance_file ../data/e-snli/train.csv --test_file sampled_instance_esnli.csv --task esnli --output_file phase1_esnli_sampled_3.csv --max_tokens 40