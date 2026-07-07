
"""train.py - build data, train, evaluate, persist."""
from src.data import make_model_data
from src.model import fit_and_evaluate
from src.evaluate import save_metrics, print_report
from src.persist import save_model

def main():
    print("Simulating transactions...")
    data=make_model_data(n=50_000)
    print("Training model...")
    model,metrics=fit_and_evaluate(data)
    save_model(model); save_metrics(metrics); print_report(metrics)
    print("\nSaved model -> models/model.pkl and metrics -> models/metrics.json")
if __name__=="__main__": main()
