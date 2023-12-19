import numpy as np
import pandas as pd
from numpy.linalg import norm


class EmbeddingUtilities:
    def __init__(self):
        pass

    @staticmethod
    def generate_cosine_similarity_df(input_embeddings_dict, label_embeddings_dict):
        """
        Calculate cosine similarity between every pair of embeddings between input and label items.

        Args:
            input_embeddings_dict (dict): Dictionary containing input item embeddings.
            label_embeddings_dict (dict): Dictionary containing label item embeddings.

        Returns:
            pd.DataFrame: DataFrame containing cosine similarity scores between input and label items.
        """
        input_embeddings_df = pd.DataFrame.from_dict(input_embeddings_dict).T.reset_index()
        input_embeddings_df.columns = ['InputText', 'InputTextEmbedding']

        label_embeddings_df = pd.DataFrame.from_dict(label_embeddings_dict).T.reset_index()
        label_embeddings_df.columns = ['LabelText', 'LabelTextEmbedding']
        mat_lab_emb0 = np.stack(label_embeddings_df['LabelTextEmbedding'])
        mat_lab_emb = np.rot90(mat_lab_emb0)[::-1]

        embedding_cross_df = pd.DataFrame()

        for row in input_embeddings_df.to_dict('records'):
            vector_doc_emb = np.stack(row['InputTextEmbedding'])

            p1 = vector_doc_emb.dot(mat_lab_emb)
            p2 = norm(mat_lab_emb, axis=0) * norm(vector_doc_emb)
            scores = p1 / p2

            data = {
                'LabelText': list(label_embeddings_df['LabelText']),
                'score': scores.tolist()
            }
            df = pd.DataFrame.from_dict(data)
            df['InputText'] = row['InputText']

            embedding_cross_df = pd.concat([embedding_cross_df, df])

        return embedding_cross_df[['InputText', 'LabelText', 'score']]

    @staticmethod
    def isolate_highest_cosine_scoring_embedding_pair(embedding_cross_df):
        """
        Isolate the highest cosine scoring embedding pairs for each 'InputText'.

        Args:
            embedding_cross_df (pd.DataFrame): DataFrame containing cosine similarity scores between input and label items.

        Returns:
            pd.DataFrame: DataFrame with the highest scoring embedding pairs for each 'InputText'.
        """
        # Select relevant columns
        col_list = ['InputText', 'score']
        cross_df = embedding_cross_df[col_list].copy()

        # Rank the scores for each 'InputText' group
        cross_df['max_score_rank'] = cross_df.groupby('InputText')['score'].rank(method='first', na_option='bottom', ascending=False)

        # Rename columns
        cross_df.rename(columns={'score': 'MaxScore'}, inplace=True)

        # Only select the highest scoring pair for each 'InputText', even in case of ties
        cross_df = cross_df[cross_df['max_score_rank'] == 1].drop(columns=['max_score_rank'])

        # Merge with the original embedding_cross_df to get complete embedding details
        top_embedding_cross_df = pd.merge(
            cross_df,
            embedding_cross_df,
            how='left',
            left_on=['InputText', 'MaxScore'], right_on=['InputText', 'score']
        ).sort_values('MaxScore', ascending=False)

        # Reduce precision
        top_embedding_cross_df['MaxScore'] = round(top_embedding_cross_df['MaxScore'],3)

        return top_embedding_cross_df[['InputText', 'LabelText', 'MaxScore']]

    @staticmethod
    def append_extra_column_info(input_data, input_text_col, db_input, db_text_col, embedding_xdf):
        """
        Append additional column information from Dataframe/JSON data to an embedding DataFrame.

        Args:
            input_data (list or pd.DataFrame): List of input JSON data or DataFrame.
            input_text_col (str): Column name containing text identifiers in the input data.
            db_input (list or pd.DataFrame): List of database JSON data or DataFrame.
            db_text_col (str): Column name containing text identifiers in the database data.
            embedding_xdf (pd.DataFrame): Embedding DataFrame to which the information will be appended.

        Returns:
            pd.DataFrame: DataFrame with additional column information appended.
        """
        # Convert input to DataFrame if it's in JSON format
        if isinstance(input_data, list):
            input_json_df = pd.DataFrame(input_data)
        else:
            input_json_df = input_data

        # Convert database input to DataFrame if it's in JSON format
        if isinstance(db_input, list):
            db_json_df = pd.DataFrame(db_input)
        else:
            db_json_df = db_input

        # Merge with the embedding DataFrame
        input_result_df = pd.merge(
            embedding_xdf,
            input_json_df,
            how='left',
            left_on='InputText',
            right_on=input_text_col
        )
        embedding_cross_detail_df = pd.merge(
            input_result_df,
            db_json_df,
            how='left',
            left_on='LabelText',
            right_on=db_text_col
        ).drop_duplicates()

        embedding_cross_detail_df.rename(columns={
             'MaxScore': 'MaxLabelsCosineScore',
             }, inplace=True)

        return embedding_cross_detail_df