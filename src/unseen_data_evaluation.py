def run_pipeline_on_unseen_data(subject_ids ,client):
  """
  Run your full pipeline, from data loading to prediction.

  :param subject_ids: A list of subject IDs of an unseen test set.
  :type subject_ids: List[int]

  :param client: A BigQuery client object for accessing the MIMIC-III dataset.
  :type client: google.cloud.bigquery.client.Client

  :return: DataFrame with the following columns:
              - subject_id: Subject IDs, which in some cases can be different due to your analysis.
              - mortality_proba: Prediction probabilities for mortality.
              - prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
              - readmission_proba: Prediction probabilities for readmission.
  :rtype: pandas.DataFrame
  """
  raise NotImplementedError('You need to implement this function')