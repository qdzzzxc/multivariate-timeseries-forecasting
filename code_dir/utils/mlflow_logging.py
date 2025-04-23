import mlflow

def log_leaderboard_to_mlflow(leaderboard_df, prefix):
    leaderboard_df = leaderboard_df.copy().drop(columns = 'fit_order')
    
    for _, row in leaderboard_df.iterrows():
        model_name = row['model']
        run_name = f"{model_name}_{prefix}"
        
        with mlflow.start_run(run_name=run_name):
            metrics_dict = row.to_dict()
            del metrics_dict['model']

            mlflow.log_metrics(metrics_dict)
            mlflow.log_param("model_name", model_name)

            mlflow.set_tag("prefix", prefix)