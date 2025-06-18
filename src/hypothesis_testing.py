import pandas as pd
from scipy import stats

def perform_ab_test(df: pd.DataFrame, group_col: str, metric_col: str, test_type='t-test'):
    groups = df.groupby(group_col)[metric_col]
    
    if test_type == 't-test':
        # Independent t-test for 2 groups
        if len(groups) == 2:
            group1, group2 = list(groups)
            return stats.ttest_ind(group1[1], group2[1])
    
    elif test_type == 'anova':
        # ANOVA for multiple groups
        return stats.f_oneway(*[g[1] for g in groups])
    
    elif test_type == 'chi2':
        # Chi-square test for categorical outcomes
        contingency = pd.crosstab(df[group_col], df[metric_col])
        return stats.chi2_contingency(contingency)
    
    return None

def risk_difference_analysis(df: pd.DataFrame):
    results = {}
    
    # Province risk differences (ANOVA)
    results['provinces'] = perform_ab_test(
        df, 'Province', 'LossRatio', 'anova'
    )
    
    # Gender risk differences (t-test)
    gender_df = df[df['Gender'].isin(['Male', 'Female'])]
    results['gender'] = perform_ab_test(
        gender_df, 'Gender', 'LossRatio', 't-test'
    )
    
    return results