import random
import string
import pandas as pd
from google.cloud import bigquery

random.seed(1236)

# Generate a uuid of a specifed length(default=8)
def generate_uuid(length: int = 8) -> str:
    return "".join(random.choices(string.ascii_lowercase + string.digits, k=length))

def download_bq_table(bq_table_uri: str) -> pd.DataFrame:
  """This function downloads a BigQuery table and returns a dataframe

  Args:
      bq_table_uri (str): BigQuery uri of the table

  Returns:
      pd.DataFrame: table data in dataframe format
  """
  # Remove bq:// prefix if present
  prefix = "bq://"
  if bq_table_uri.startswith(prefix):
      bq_table_uri = bq_table_uri[len(prefix) :]

  client = bigquery.Client()

  table = client.get_table(bq_table_uri)
  rows = client.list_rows(table)

  return rows.to_dataframe()


def save_dataframe_to_bigquery(
  dataframe: pd.DataFrame,
  table_name: str
  ) -> str:
  """This function loads a dataframe to a new bigquery table

  Args:
      dataframe (pd.Dataframe): dataframe to be loaded to bigquery
      table_name (str): name of the bigquery table that is being created

  Returns:
      str: table id of the destination bigquery table
  """
  client = bigquery.Client()
  project_id = client.project
  dataset_id = generate_uuid()

  bq_dataset = bigquery.Dataset(f"{project_id}.{dataset_id}")
  bq_dataset = client.create_dataset(bq_dataset, exists_ok=True)

  job_config = bigquery.LoadJobConfig(
        # Specify a (partial) schema. All columns are always written to the
        # table. The schema is used to assist in data type definitions.
        schema=[
            bigquery.SchemaField("date", bigquery.enums.SqlTypeNames.DATE),
        ],
        # Optionally, set the write disposition. BigQuery appends loaded rows
        # to an existing table by default, but with WRITE_TRUNCATE write
        # disposition it replaces the table with the loaded data.
        write_disposition="WRITE_TRUNCATE",
    )

  # Reference: https://cloud.google.com/bigquery/docs/samples/bigquery-load-table-dataframe
  job = client.load_table_from_dataframe(
      dataframe=dataframe,
      destination=f"{project_id}.{dataset_id}.{table_name}",
      job_config=job_config,
  )

  job.result()

  return str(job.destination)


def rename_bq_table_column(
  table_id: str,
  old_column_name: str,
  new_column_name: str
  ) -> bigquery.QueryJob:
  """This function renames a column name in a bigquery table.

  Args:
      table_id (str): Id of the table in which the column is renamed
      old_column_name (str): The current name of the column to be renamed
      new_column_name (str): The new column name

  Returns:
      bigquery.QueryJob: The query job of the altering the table
  """
  query = f"ALTER TABLE {table_id} \
        RENAME COLUMN {old_column_name} TO {new_column_name};"

  client = bigquery.Client()
  job = client.query(query=query)

  return job




