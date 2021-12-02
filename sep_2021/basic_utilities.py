import pandas as pd
import numpy as np 
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import quantile_transform
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import json


ERA_COL = "era"
TARGET_COL = "target"
EXAMPLE_PRED_COL = 'ex_preds'
CRED = '\033[31m'
CGREEN = '\033[32m'
CEND = '\033[0m'
CBOLD = '\033[1m'
G_SPACE = 30
I_SPACE = 19
R_SPACE = G_SPACE - I_SPACE
N_SPACE = 7


# Some intervals are not up to date
VALIDATION_METRIC_INTERVALS = {
    "mean": (0.013, 0.028),
    "sharpe": (0.53, 1.24),
    "std": (0.0303, 0.0168),
    "max_feature_exposure": (0.4, 0.0661),
    "mmc_mean": (-0.008, 0.008),
    "corr_plus_mmc_sharpe": (0.41, 1.34),
    "max_drawdown": (-0.115, -0.025),
    "feature_neutral_mean": (0.006, 0.022)
} 


# Create new `pandas` methods which use `tqdm` progress
# (can use tqdm_gui, optional kwargs, etc.)
tqdm.pandas()

training_features_dict = json.load(open('training_features_dict.json'))

def color_metric(metric_value, metric_name):
    low, high = VALIDATION_METRIC_INTERVALS[metric_name]
    pct = stats.percentileofscore(np.linspace(low, high, 100),
                                  metric_value)
    if high <= low:
        pct = 100 - pct
    if pct > 95:  # Excellent
        return CGREEN
    elif pct > 75:  # Good
        return CGREEN
    elif pct > 35:  # Fair
        return CEND
    else:  # Bad
        return CRED



def autocorr(x, t=1):
    return np.corrcoef([x[:-t], x[t:]])[0,1]

def get_riskiest_features(df, target='target',features=None, group='era', num=50):
    if features == None:
        features = [x for x in df.columns if x.startswith('feature')]
    feature_corr_volatility = pd.Series(training_features_dict['feature_corr_volatility'])[features]
    feature_exposure_list = []
    for feature in tqdm(features):
        feature_exposure_list.append(np.corrcoef(df[feature], df[target])[0,1])
    feature_exposure_list = pd.Series(feature_exposure_list, index=features)
    return (feature_exposure_list.abs()*feature_corr_volatility).sort_values(ascending = False).head(num).index.to_list()

# def get_riskiest_features(df, target='target',features=None, group='era', num=50):
#     if features == None:
#         features = [x for x in df.columns if x.startswith('feature')]
#     all_feature_corrs=df.groupby(group).progress_apply(lambda x: x[features].corrwith(x[target]))
#     feature_corr_volatility = all_feature_corrs.std(axis=0)
#     return (all_feature_corrs.mean(axis=0).abs()*feature_corr_volatility).sort_values(ascending = False).head(num).index.to_list()


def neutralize(df, target="target", by=None, proportion=1.0):
    if by is None:
        by = [x for x in df.columns if x.startswith('feature')]

    scores = df[target]
    exposures = df[by].values

    # constant column to make sure the series is completely neutral to exposures
    exposures = np.hstack((exposures, np.array([np.mean(scores)] * len(exposures)).reshape(-1, 1)))

    scores -= proportion * (exposures @ (np.linalg.pinv(exposures) @ scores.values))
    return scores / scores.std()


def full_neutralization(df, feature_names, pred_name="target", proportion=1.0):
    new_target = df.groupby("era").progress_apply(lambda x: neutralize(x, pred_name, feature_names, proportion))
    scaled_preds = MinMaxScaler().fit_transform(new_target.to_frame())
    return scaled_preds


def neutralize_series(series, by, proportion=1.0):
    scores = series.values.reshape(-1, 1)
    exposures = by.values.reshape(-1, 1)

    # this line makes series neutral to a constant column so that it's centered and for sure gets corr 0 with exposures
    exposures = np.hstack(
        (exposures,
         np.array([np.mean(series)] * len(exposures)).reshape(-1, 1)))

    correction = proportion * (exposures.dot(
        np.linalg.lstsq(exposures, scores, rcond=None)[0]))
    corrected_scores = scores - correction
    neutralized = pd.Series(corrected_scores.ravel(), index=series.index)
    return neutralized

def unif(df):
    #x = quantile_transform(df.values.reshape(-1,1), axis=0,  output_distribution='uniform', copy=True).ravel()
    x = (df.rank(method="first")) / len(df)
    return pd.Series(x, index=df.index)

def normif(df):
    x = quantile_transform(df.values.reshape(-1,1), axis=0,  output_distribution='normal', 
                           subsample=len(df), random_state=68, copy=True).ravel()
    return pd.Series(x, index=df.index)

def get_feature_neutral_mean(df, prediction_col):
    feature_cols = [c for c in df.columns if c.startswith("feature")]
    df['sub'] = unif(df[prediction_col])
    df.loc[:, "neutral_sub"] = full_neutralization(df, feature_cols, pred_name='sub', proportion=1.0)
    # neutralize(df, [prediction_col],
    #                                       feature_cols)
    df = df.drop('sub', axis=1)
    scores = df.groupby("era").progress_apply(
        lambda x: ((x["neutral_sub"]).corr(x[TARGET_COL]))).mean()
    return np.mean(scores)

def calculate_fnc(pred_name, target, df):
    """    
    Args:
        pred_name (str)
        target (str)
        df (pd.DataFrame)
    """
    # Normalize submission
    df['sub'] = normif(df[pred_name])
    # Neutralize submission to features
    sub = full_neutralization(df,  feature_names=None, pred_name='sub', proportion=1.0)
    #Drop tmp column
    df = df.drop('sub', axis=1)
    sub = pd.Series(np.squeeze(sub)) # Convert np.ndarray to pd.Series
    # FNC: Spearman rank-order correlation of neutralized submission to target
    fnc = np.corrcoef(sub.rank(pct=True, method="first"), df[target])[0, 1]
    return fnc

def fast_score_by_date(df, columns, target, tb=None, era_col="era"):
    unique_eras = df[era_col].unique()
    computed = []
    for u in tqdm(unique_eras):
        df_era = df[df[era_col] == u]
        era_pred = np.float64(df_era[columns].values.T)
        era_target = np.float64(df_era[target].values.T)

        if tb is None:
            ccs = np.corrcoef(era_target, era_pred)[0, 1:]
        else:
            tbidx = np.argsort(era_pred, axis=1)
            tbidx = np.concatenate([tbidx[:, :tb], tbidx[:, -tb:]], axis=1)
            ccs = [np.corrcoef(era_target[tmpidx], tmppred[tmpidx])[0, 1] for tmpidx, tmppred in zip(tbidx, era_pred)]
            ccs = np.array(ccs)

        computed.append(ccs)

    return pd.DataFrame(np.array(computed), columns=columns, index=df[era_col].unique())


def validation_metrics(validation_data, pred_cols, example_col, fast_mode=False):
    tb200_validation_correlations = None
    validation_correlations = None
    validation_stats = pd.DataFrame()
    feature_cols = [c for c in validation_data if c.startswith("feature_")]
    for pred_col in pred_cols:
        print('Checking the per-era correlations on the validation set (out of sample)')
        # Check the per-era correlations on the validation set (out of sample)
        validation_correlations = validation_data.groupby(ERA_COL).progress_apply(
            lambda d: unif(d[pred_col]).corr(d[TARGET_COL]))

        mean = validation_correlations.mean()
        std = validation_correlations.std(ddof=0)
        sharpe = mean / std

        validation_stats.loc["mean", pred_col] = mean
        validation_stats.loc["std", pred_col] = std
        validation_stats.loc["sharpe", pred_col] = sharpe

        rolling_max = (validation_correlations + 1).cumprod().rolling(window=9000,  # arbitrarily large
                                                                      min_periods=1).max()
        daily_value = (validation_correlations + 1).cumprod()
        max_drawdown = -((rolling_max - daily_value) / rolling_max).max()
        validation_stats.loc["max_drawdown", pred_col] = max_drawdown

        payout_scores = validation_correlations.clip(-0.25, 0.25)
        payout_daily_value = (payout_scores + 1).cumprod()

        apy = (
            (
                (payout_daily_value.dropna().iloc[-1])
                ** (1 / len(payout_scores))
            )
            ** 49  # 52 weeks of compounding minus 3 for stake compounding lag
            - 1
        ) * 100

        validation_stats.loc["apy", pred_col] = apy

        if not fast_mode:
            print('Checking the feature exposure of your validation predictions')
            
            # Check the feature exposure of your validation predictions  
            max_per_era = validation_data.groupby(ERA_COL).apply(lambda d: d.filter(like='feature').corrwith(d[example_col]).abs().max())
            max_feature_exposure = np.mean(max_per_era)
            validation_stats.loc["max_feature_exposure", pred_col] = max_feature_exposure

            # Check feature neutral mean
            feature_neutral_mean = get_feature_neutral_mean(validation_data, pred_col)
            validation_stats.loc["feature_neutral_mean", pred_col] = feature_neutral_mean

            # Check top and bottom 200 metrics (TB200)
            tb200_validation_correlations = fast_score_by_date(
                validation_data,
                [pred_col],
                TARGET_COL,
                tb=200,
                era_col=ERA_COL
            )

            tb200_mean = tb200_validation_correlations.mean()[pred_col]
            tb200_std = tb200_validation_correlations.std(ddof=0)[pred_col]
            tb200_sharpe = tb200_mean / tb200_std

            validation_stats.loc["tb200_mean", pred_col] = tb200_mean
            validation_stats.loc["tb200_std", pred_col] = tb200_std
            validation_stats.loc["tb200_sharpe", pred_col] = tb200_sharpe
            
            tb200_validation_correlations = tb200_validation_correlations.squeeze()

        # MMC over validation
        mmc_scores = []
        corr_scores = []
        for _, x in tqdm(validation_data.groupby(ERA_COL)):
            series = neutralize_series(unif(x[pred_col]), unif(x[EXAMPLE_PRED_COL]))
            mmc_scores.append(np.cov(series, x[TARGET_COL])[0, 1] / (0.29 ** 2))
            corr_scores.append(unif(x[pred_col]).corr(x[TARGET_COL]))

        val_mmc_mean = np.mean(mmc_scores)
        val_mmc_std = np.std(mmc_scores)
        corr_plus_mmcs = [c + m for c, m in zip(corr_scores, mmc_scores)]
        corr_plus_mmc_sharpe = np.mean(corr_plus_mmcs) / np.std(corr_plus_mmcs)

        validation_stats.loc["mmc_mean", pred_col] = val_mmc_mean
        validation_stats.loc["corr_plus_mmc_sharpe", pred_col] = corr_plus_mmc_sharpe

        # Check correlation with example predictions
        per_era_corrs = validation_data.groupby(ERA_COL).progress_apply(lambda d: unif(d[pred_col]).corr(unif(d[EXAMPLE_PRED_COL])))
        corr_with_example_preds = np.mean(per_era_corrs)
        validation_stats.loc["corr_with_example_preds", pred_col] = corr_with_example_preds
        
        # Autocorrelation (not correct)
        validation_stats.loc["autocorr", pred_col] = autocorr(validation_data[pred_col])
          
        
    # .transpose so that stats are columns and the model_name is the row
    return tb200_validation_correlations,validation_correlations,validation_stats.transpose()


def make_plot(df, ax, title):
    sns.set_style("whitegrid")
    figure = sns.barplot(x=df.index, y=df, palette=['black'], ax=ax)
    sns.despine(left=True)
    ax.set_ylim(min(-0.04, min(df)-0.01),max(0.08, max(df)+0.01))
    ax.set_title(f'Validation Correlation: {title}')
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    return figure

def make_cum_plot(df, ax, title):
    sns.set_style("whitegrid")
    figure = sns.lineplot(x=df.index, y=df.cumsum(), color='black', ax=ax)
    sns.despine(left=True)
    #ax.set_ylim(min(-0.04, min(df)-0.01),max(0.08, max(df)+0.01))
    ax.set_title(f'Validation Correlation: {title} cumulative')
    ax.xaxis.set_major_locator(plt.MultipleLocator(5))
    return figure

def fmt_score(score, spacing):
    return str(round(score, 4)).rjust(spacing)

def score_line(df,title, pred_name):
    pred = df.first_valid_index()
    return f'{title.ljust(I_SPACE)}{color_metric(df.loc[pred, pred_name],pred_name)}{fmt_score(df.loc[pred, pred_name], N_SPACE).ljust(R_SPACE)}{CEND}'

def print_scores(df):
    pred = df.first_valid_index()
    # Headers
    print(f'{CBOLD}{"Performance".ljust(G_SPACE)}{"Risk".ljust(G_SPACE)}{"MMC".ljust(G_SPACE)}{"Other".ljust(G_SPACE)}{CEND}')
    print() # Empty line for better spacing
    # First line
    output = score_line(df,"Sharpe ratio", "sharpe")
    output += score_line(df,"Std. Dev.", "std")
    output += score_line(df,"Corr + MMC Sharpe", "corr_plus_mmc_sharpe")
    output += f'{"APY".ljust(I_SPACE)}{str(round(df.loc[pred, "apy"],4)).rjust(R_SPACE)}{CEND}'
    print(output)
    # Second line
    output = score_line(df,"Corr.", "mean")
    output += score_line(df,"Feat. Exposure", "max_feature_exposure")
    output += score_line(df,"MMC Mean", "mmc_mean")
    output += f'{"Autocorr.".ljust(I_SPACE)}{str(round(df.loc[pred, "autocorr"],4)).rjust(R_SPACE)}'
    print(output)
    # Third line
    output = score_line(df,"FNC.", "feature_neutral_mean")
    output += score_line(df,"Max DrawDown", "max_drawdown")
    output += f'{"Ex. Preds Corr".ljust(I_SPACE)}{str(round(df.loc[pred, "corr_with_example_preds"],4)).rjust(N_SPACE).ljust(R_SPACE)}'
    print(output)
    print() # Empty line for better spacing
    return
    
    
