# src/travel_ai/bn/bn_examples.py
import json
import os
from bn_model import build_disruption_cpt, save_cpt_csv, infer_pure_python

def pretty_print_post(post):
    print("Posterior DisruptionRisk:")
    for k,v in post.items():
        print(f"  {k:6s}: {v:.4f}")

def main():
    cpt = build_disruption_cpt()
    save_cpt_csv(cpt, path='outputs/disruption_cpt.csv')

    # Non-trivial test case (realistic): severe weather, high history delay, high congestion, no festival
    evidence = {'Weather':2,'HistoryDelay':1,'Congestion':1,'Festival':0}
    print("Test case evidence:", evidence)
    posterior = infer_pure_python(evidence, cpt=cpt)
    pretty_print_post(posterior)

    # Save posterior for PDF
    os.makedirs('outputs', exist_ok=True)
    with open('outputs/posterior_example.json', 'w') as f:
        json.dump({'evidence': evidence, 'posterior': posterior}, f, indent=2)
    print("Saved posterior to outputs/posterior_example.json")

if __name__ == "__main__":
    main()
