import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, hamming_loss, roc_curve, auc,
    roc_auc_score, precision_recall_curve, classification_report
)


class ClassificationEvaluator:
    def __init__(self, align_doc_label_df, align_doc_id_column_name,
                 label_df, label_column_name):
        """
        Initialize the ClassificationEvaluator.

        Parameters:
        - align_doc_label_df: Validation dataset with labels.
        - align_doc_id_column_name: Column name specific to align_doc_label_df, e.g., 'doc_id'.
        - label_df: Dataframe that contains the labels.
        - label_column_name: Column name specific to label_df.
        """
        self.align_doc_label_df = align_doc_label_df
        self.align_doc_id_column_name = align_doc_id_column_name
        self.label_df = label_df
        self.label_column_name = label_column_name

    def create_lists_of_docs_and_labels(self):
        """
        Create lists of unique document IDs and labels.

        Returns:
        - doc_id_list: List of unique document IDs.
        - label_text_list: List of unique labels.
        """
        doc_id_list = list(self.align_doc_label_df[self.align_doc_id_column_name].unique())
        label_text_list = list(self.label_df[self.label_df[self.label_column_name].notna()
                                            ][self.label_column_name].unique())
        num_docs = len(doc_id_list)
        num_labels = len(label_text_list)
        print(f'Number of Unique Docs: {num_docs}')
        print(f'Number of Unique Labels: {num_labels}')
        return doc_id_list, label_text_list

    def cross_docs_and_labels(self, doc_id_list, label_text_list,
                              align_doc_label_df_subset, align_doc_id_column_name,
                              align_doc_label_column_name):
        """
        Generate a cross-tabulation of document IDs and labels, including a boolean ground truth field.

        Parameters:
        - doc_id_list: List of unique document IDs.
        - label_text_list: List of unique labels.
        - align_doc_label_df_subset: Subset of the validation dataset with labels.
        - align_doc_id_column_name: Column name specific to align_doc_label_df_subset, e.g., 'parent_doc_id'.
        - align_doc_label_column_name: Column name specific to align_doc_label_df_subset.

        Returns:
        - esd: DataFrame with columns 'doc_id', 'label_text', and 'is_actual_value'.
        """
        cross_doc_labels = itertools.product(doc_id_list, label_text_list)
        cx_df = pd.DataFrame(list(cross_doc_labels))
        cx_df.rename(columns={0: 'doc_id', 1: 'label_text'}, inplace=True)
        cx_df_plus_truth = pd.merge(
            cx_df,
            align_doc_label_df_subset[[align_doc_id_column_name, align_doc_label_column_name]].drop_duplicates(),
            how='left',
            left_on=['doc_id', 'label_text'],
            right_on=[align_doc_id_column_name, align_doc_label_column_name]
        )
        cx_df_plus_truth['is_actual_value'] = cx_df_plus_truth[align_doc_label_column_name].notna().astype(int)
        esd = cx_df_plus_truth[['doc_id', 'label_text', 'is_actual_value']].copy()
        return esd

    @staticmethod
    def convert_df_to_wide_and_numpy(df, identifier_col_name, category_col_name, value_col_name):
        """
        Convert a DataFrame to wide format and return the result along with category list and NumPy array.

        Parameters:
        - df: Input DataFrame.
        - identifier_col_name: Column name to be used as an identifier, e.g., 'id'.
        - category_col_name: Column name containing categories.
        - value_col_name: Column name containing values to be pivoted.

        Returns:
        - df_wide_values: DataFrame in wide format.
        - df_category_list: List of category names.
        - df_wide_values_numpy: NumPy array representation of the wide DataFrame.
        """
        df_wide_values = pd.pivot_table(
            df,
            values=value_col_name,
            index=[identifier_col_name],
            columns=[category_col_name],
            aggfunc=np.max, fill_value=0
        ).reset_index()
        df_wide_values.drop(columns=identifier_col_name, inplace=True)
        df_category_list = list(df_wide_values.columns)
        df_wide_values_numpy = df_wide_values.to_numpy()
        return df_wide_values, df_category_list, df_wide_values_numpy

    def create_actual_predictions_table(self, actual_doc_labels_cx_df,
                                              pred_doc_label_df, pred_doc_id_column_name,
                                              pred_doc_label_column_name, pred_doc_score_column_name,
                                              threshold_val=0):
        """
        Create a table with actual and predicted labels based on a threshold value.

        Parameters:
        - actual_doc_labels_cx_df: DataFrame containing ground truth labels.
        - pred_doc_label_df: DataFrame containing predicted labels.
        - pred_doc_id_column_name: Column name for document IDs in the predicted DataFrame.
        - pred_doc_label_column_name: Column name for labels in the predicted DataFrame.
        - pred_doc_score_column_name: Column name for prediction scores in the predicted DataFrame.
        - threshold_val: Threshold value for considering predicted labels.

        Returns:
        - actual_doc_labels_cx_plus_pred: DataFrame containing actual and predicted labels.
        """
        actual_doc_labels_cx_plus_pred = pd.merge(
            actual_doc_labels_cx_df,
            pred_doc_label_df[(pred_doc_label_df[pred_doc_score_column_name] > threshold_val)][
                [pred_doc_id_column_name, pred_doc_label_column_name, pred_doc_score_column_name]].drop_duplicates(),
            how='left',
            left_on=['doc_id', 'label_text'],
            right_on=[pred_doc_id_column_name, pred_doc_label_column_name]
        )
        actual_doc_labels_cx_plus_pred.drop(columns=[pred_doc_id_column_name, pred_doc_label_column_name], inplace=True)
        actual_doc_labels_cx_plus_pred.fillna(0, inplace=True)
        return actual_doc_labels_cx_plus_pred

    def prepare_data_for_classification_report(self, actual_doc_labels_cx_plus_pred, 
                                              pred_doc_score_column_name):
        """
        Prepare data for generating a classification report.

        Parameters:
        - actual_doc_labels_cx_plus_pred: DataFrame containing actual and predicted labels.
        - pred_doc_score_column_name: Column name for prediction scores in the predicted DataFrame.

        Returns:
        - df_label_list: List of category names.
        - actual_df_wide_values_numpy: NumPy array representation of the actual labels in wide format.
        - predicted_df_wide_values_numpy: NumPy array representation of the predicted labels in wide format.
        """
        _, df_label_list, actual_df_wide_values_numpy = self.convert_df_to_wide_and_numpy(
            df=actual_doc_labels_cx_plus_pred,
            identifier_col_name='doc_id',
            category_col_name='label_text',
            value_col_name='is_actual_value')

        _, _, predicted_df_wide_values_numpy = self.convert_df_to_wide_and_numpy(
            df=actual_doc_labels_cx_plus_pred,
            identifier_col_name='doc_id',
            category_col_name='label_text',
            value_col_name=pred_doc_score_column_name)

        return df_label_list, actual_df_wide_values_numpy, predicted_df_wide_values_numpy

    def create_classification_confusion_table(self, actual_doc_labels_cx_df,
                                              pred_doc_label_df, pred_doc_id_column_name,
                                              pred_doc_label_column_name, pred_doc_score_column_name,
                                              threshold_val=0):
        """
        Create a classification confusion table and additional metrics.

        Parameters:
        - actual_doc_labels_cx_df: DataFrame containing ground truth labels.
        - pred_doc_label_df: DataFrame containing predicted labels.
        - pred_doc_id_column_name: Column name for document IDs in the predicted DataFrame.
        - pred_doc_label_column_name: Column name for labels in the predicted DataFrame.
        - pred_doc_score_column_name: Column name for prediction scores in the predicted DataFrame.
        - threshold_val: Threshold value for considering predicted labels.

        Returns:
        - cdf_long_plus_adj: DataFrame containing classification report metrics and additional metrics.
        - num_labels_per_doc_df: DataFrame with the number of labels per document.
        """

        actual_doc_labels_cx_plus_pred = self.create_actual_predictions_table(
            actual_doc_labels_cx_df,
            pred_doc_label_df,
            pred_doc_id_column_name,
            pred_doc_label_column_name,
            pred_doc_score_column_name,
            threshold_val=0
        )

        df_label_list, actual_df_wide_values_numpy, predicted_df_wide_values_numpy = self.prepare_data_for_classification_report(
            actual_doc_labels_cx_plus_pred,
            pred_doc_score_column_name
        )

        clf = classification_report(y_true=actual_df_wide_values_numpy,
                                    y_pred=(predicted_df_wide_values_numpy > threshold_val) * 1.0,
                                    target_names=df_label_list,
                                    zero_division=0,
                                    output_dict=True)
        print('\nHamming Loss:', hamming_loss(actual_df_wide_values_numpy,
                                              (predicted_df_wide_values_numpy > threshold_val) * 1.0))
        clfdf = pd.DataFrame(clf)
        cdf = clfdf.T
        for col in ['precision', 'recall', 'f1-score']:
            cdf[col] = cdf[col].map('{:,.2f}'.format)
        cdf_long = cdf.reset_index()
        cdf_long.rename(columns={'index': 'label_text'}, inplace=True)
        fndf = actual_doc_labels_cx_plus_pred[(actual_doc_labels_cx_plus_pred[pred_doc_score_column_name] == 0) &
                                              (actual_doc_labels_cx_plus_pred['is_actual_value'] == 1)
                                              ].groupby('label_text')['doc_id'].nunique().reset_index()
        fndf.columns = ['label_text', 'num false negatives']
        fpdf = actual_doc_labels_cx_plus_pred[(actual_doc_labels_cx_plus_pred[pred_doc_score_column_name] > 0) &
                                              (actual_doc_labels_cx_plus_pred['is_actual_value'] == 0)
                                              ].groupby('label_text')['doc_id'].nunique().reset_index()
        fpdf.columns = ['label_text', 'num false positives']
        tpdf = actual_doc_labels_cx_plus_pred[(actual_doc_labels_cx_plus_pred[pred_doc_score_column_name] > 0) &
                                              (actual_doc_labels_cx_plus_pred['is_actual_value'] == 1)
                                              ].groupby('label_text')['doc_id'].nunique().reset_index()
        tpdf.columns = ['label_text', 'num true positives']
        tndf = actual_doc_labels_cx_plus_pred[(actual_doc_labels_cx_plus_pred[pred_doc_score_column_name] == 0) &
                                              (actual_doc_labels_cx_plus_pred['is_actual_value'] == 0)
                                              ].groupby('label_text')['doc_id'].nunique().reset_index()
        tndf.columns = ['label_text', 'num true negative']

        cdf_long_plus = pd.merge(cdf_long, fpdf, how='left')
        cdf_long_plus = pd.merge(cdf_long_plus, fndf, how='left')
        cdf_long_plus = pd.merge(cdf_long_plus, tpdf, how='left')
        cdf_long_plus = pd.merge(cdf_long_plus, tndf, how='left')

        overall_metrics = cdf_long_plus.tail(4).copy()
        overall_metrics['num false negatives'] = cdf_long_plus['num false negatives'].sum()
        overall_metrics['num false positives'] = cdf_long_plus['num false positives'].sum()
        overall_metrics['num true positives'] = cdf_long_plus['num true positives'].sum()
        overall_metrics['num true negative'] = cdf_long_plus['num true negative'].sum()

        overall_metrics_percent = cdf_long_plus.tail(2).copy()

        total_options = (cdf_long_plus['num false negatives'].sum() + cdf_long_plus['num false positives'].sum() +
                         cdf_long_plus['num true positives'].sum() + cdf_long_plus['num true negative'].sum())
        overall_metrics_percent['num false negatives'] = round(
            cdf_long_plus['num false negatives'].sum() / total_options, 2)
        overall_metrics_percent['num false positives'] = round(
            cdf_long_plus['num false positives'].sum() / total_options, 2)
        overall_metrics_percent['num true positives'] = round(
            cdf_long_plus['num true positives'].sum() / total_options, 2)
        overall_metrics_percent['num true negative'] = round(
            cdf_long_plus['num true negative'].sum() / total_options, 2)

        cdf_long_plus_adj = pd.concat([
            cdf_long_plus[:-4],
            overall_metrics[-4:-2],
            overall_metrics_percent]
        )

        cdf_long_plus_adj.fillna(0, inplace=True)

        num_labels_per_doc_df = actual_doc_labels_cx_plus_pred[
            (actual_doc_labels_cx_plus_pred[pred_doc_score_column_name] > 0)
        ].groupby('doc_id')['label_text'].nunique().reset_index()
        num_labels_per_doc_df.rename(columns={'label_text': 'NumLabels'}, inplace=True)
        num_labels_per_doc_df.sort_values('NumLabels', ascending=False, inplace=True)

        return cdf_long_plus_adj, num_labels_per_doc_df, actual_doc_labels_cx_plus_pred