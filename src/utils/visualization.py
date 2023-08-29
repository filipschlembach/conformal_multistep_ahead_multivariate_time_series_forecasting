import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# todo: refactor and integrate into the rest of the codebase

class Visualization:

    @staticmethod
    def plot_data_set(data_set_df: pd.DataFrame, data_col: str, prediction_col: str,
                      cal_test_split_idx: int = None) -> go.Figure:
        """
        Plots the time series and the model's prediction. Adds vertical line for calibration test split if given
        :param data_set_df: data set containing a column for the time series and one for the predictions
        :param data_col: name of the time series column
        :param prediction_col: name of the prediction column
        :param cal_test_split_idx: index of the first element in the test set if split along the time axis
        :return: go.Figure
        """
        fig = px.line(data_set_df, y=[data_col, prediction_col])
        if cal_test_split_idx is not None:
            fig.add_vline(x=cal_test_split_idx, line_width=1, line_color="black",
                          annotation={'text': 'calibration (left), validation (right) split'})
        fig.update_layout(title="Data Set")
        return fig

    @staticmethod
    def plot_data_set_and_residuals(data_set_df: pd.DataFrame, data_col: str, prediction_col: str, residual_col: str,
                                    cal_test_split_idx: int = None, cal_test_split_col: str = None) -> go.Figure:
        """
        Plots the time series and the model's prediction. Adds vertical line for calibration test split if given
        :param data_set_df: data set containing a column for the time series and one for the predictions
        :param data_col: name of the time series column
        :param prediction_col: name of the prediction column
        :param residual_col: name of the residual column
        :param cal_test_split_idx: index of the first element in the test set if split along the time axis
        :param cal_test_split_col: column used to color the different sections of the data set
        :return: go.Figure
        """
        if cal_test_split_col is not None:
            line_fig = px.line(data_set_df, y=[data_col, prediction_col], color=cal_test_split_col)
        else:
            line_fig = px.line(data_set_df, y=[data_col, prediction_col])
        fig = go.Figure(line_fig.data + px.bar(data_set_df, y=['residuals']).data)
        if cal_test_split_idx is not None:
            fig.add_vline(x=cal_test_split_idx, line_width=1, line_color="black",
                          annotation={'text': 'calibration (left), validation (right) split'})
        fig.update_layout(title="Data Set and Residuals")
        return fig

    @staticmethod
    def plot_empiric_quantile(data_series: pd.Series, quantile: float, title: str = None) -> go.Figure:
        """
        Shows the empiric epsilons, computed based on the index.
        :param data_series: displayed data
        :param quantile: percentile
        :param title: alternative title
        :return:
        """
        fig = px.bar(data_series)
        fig.add_vline(x=int(len(data_series) * quantile), line_width=1, line_color="black",
                      annotation={'text': f'{quantile * 100} %ile'})
        if title is not None:
            fig.update_layout(title=title)
        else:
            fig.update_layout(title="Sorted Residuals and Empiric Quantile")
        return fig

    @staticmethod
    def plot_rediciton_interval(data_set_df: pd.DataFrame, data_col: str, prediction_col: str, lower_col: str,
                                upper_col: str, cert: float, cal_test_split_idx: int = None,
                                cal_test_split_col: str = None) -> go.Figure:
        """
        Plots the data, prediction and confidence interval
        :param data_set_df: data set containing a column for the time series and one for the predictions
        :param data_col: name of the time series column
        :param prediction_col: name of the prediction column
        :param lower_col: name of the column containing the lower bound of the confidence interval
        :param upper_col:name of the column containing the upper bound of the confidence interval
        :param cert: 1 - alpha
        :param cal_test_split_idx: index of the first element in the test set if split along the time axis
        :param cal_test_split_col: column used to color the different sections of the data set
        :return: go.Figure
        """
        if cal_test_split_col is not None:
            fig = px.line(data_set_df, y=[data_col, prediction_col], color=cal_test_split_col)
        else:
            fig = px.line(data_set_df, y=[data_col, prediction_col], color='variable')
        fig.add_trace(go.Scatter(
            x=np.concatenate([data_set_df.index.values, data_set_df.index.values[::-1]]),
            y=np.concatenate([data_set_df.loc[:, upper_col].values, data_set_df.loc[:, lower_col].values[::-1]]),
            fill='toself',
            fillcolor='rgba(0,100,80,0.2)',
            line_color='rgba(255,255,255,0)',
            name=f'{cert * 100} % prediction interval',
        ))
        if cal_test_split_idx is not None:
            fig.add_vline(x=cal_test_split_idx, line_width=1, line_color="black",
                          annotation={'text': 'calibration (left), validation (right) split'})
        fig.update_layout(title="Data Set and Prediction Interval")
        return fig
