import json

from src.models.trainer import CreditModelTrainer


def main() -> None:
    trainer = CreditModelTrainer()
    _, metrics, global_shap = trainer.run_training_pipeline()
    output = {
        "metrics": metrics.model_dump(),
        "top_5_shap_features": dict(list(global_shap.items())[:5]),
    }
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
