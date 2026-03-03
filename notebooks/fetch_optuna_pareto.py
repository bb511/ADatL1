import optuna
import pandas as pd

# Use the SAME storage URL as the dashboard
study = optuna.load_study(
    study_name="cap_vs_loss_bs16k_v3",
    storage="sqlite:////data/deodagiu/adl1t/logs/optuna/ae_fresh.db"
)

df = study.trials_dataframe(attrs=("number", "values", "params", "state"))
# Optuna’s Pareto front (multi-objective)
pareto_numbers = {t.number for t in study.best_trials}

df["is_pareto"] = df["number"].isin(pareto_numbers)

df.to_csv("notebooks/paretos/cap_vs_loss_bs16k_v3.csv", index=False)

print("total:", len(df), "pareto:", df["is_pareto"].sum())
print("directions:", study.directions)
