#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import matplotlib.pyplot as plt
#%%

# read data
df_AB = pd.read_csv("df_AB_all_accuracies_run1.csv")
df_AC = pd.read_csv("df_AC_all_accuracies_run1.csv")
df_loss_accuracies = pd.read_csv("loss_df_run1.csv")


# prepare data for analysis
df_AB.insert(0, "condition", ["A&B"] * len(df_AB))
df_AC.insert(0, "condition", ["A&C"] * len(df_AC))
df_concat = pd.concat([df_AB, df_AC], axis = 0)
df_concat.insert(0, "accuracy", list(df_loss_accuracies["accuracy"]) * 2)
df_concat.insert(0, "loss", list(df_loss_accuracies["loss"]) * 2)

# for analysis filter rows where accuracy is greater or equal than 0.9 = 90%
df_analysis = df_concat.loc[(df_concat["loss"]  <  0.01)]
df_analysis = df_analysis.loc[:, ~df_analysis.columns.str.startswith('Un')]
df_analysis = df_analysis.loc[:, ~df_analysis.columns.str.startswith('in')]
df_analysis = df_analysis.rename(columns={"val_acc_cuepos0" : "0" ,
                                          "val_acc_cuepos1" : "1",	
                                          "val_acc_cuepos2" : "2",	
                                          "val_acc_cuepos3" : "3",	
                                          "val_acc_cuepos4" : "4",	
                                          "val_acc_cuepos5" : "5",	
                                          "val_acc_cuepos6" : "6",	
                                          "val_acc_cuepos7" : "7",	
                                          "val_acc_cuepos8" : "8",	
                                          "val_acc_cuepos9" : "9"})
df_analysis = pd.melt(df_analysis, id_vars='condition', value_vars=["0","1","2","3","4","5","6","7","8","9"])
df_analysis = df_analysis.rename(columns={"variable" : "SOA_of_cue", "value" : "accuracy"})
#%%
# (Two-Way ANOVA (Typ III)) .. levene test ist missing
model = ols('accuracy ~ C(condition) + C(SOA_of_cue) + C(condition):C(SOA_of_cue)', data=df_analysis).fit()
sm.stats.anova_lm(model, typ=3)
#%%
# Two-sample t-test (Welch)
stats.ttest_ind(a=list(df_analysis.loc[df_analysis["condition"] == "A&B"]["accuracy"]), b=list(df_analysis.loc[df_analysis["condition"] == "A&C"]["accuracy"]), equal_var= False)
# Calculate variance for group A&B
variance_AB = np.var(df_analysis.loc[df_analysis["condition"] == "A&B"]["accuracy"])
print("Variance for A&B:", variance_AB)

# Calculate variance for group A&C
variance_AC = np.var(df_analysis.loc[df_analysis["condition"] == "A&C"]["accuracy"])
print("Variance for A&C:", variance_AC)

# %%
# create boxplot
boxplot_0 = df_analysis.boxplot(column=['accuracy'], by='condition', grid=False, color='black')
# Set the title of the boxplot and remove the default title
title_boxplot_1 = "Accuracy Values Grouped by Condition"
plt.title(title_boxplot_1)
plt.suptitle("")
boxplot_0.set_xlabel("Condition")
boxplot_0.set_ylabel("Accuracy")# %%

# np.var(df_analysis.loc[df_analysis["condition"] == "A&B"]["accuracy"])
# np.var(df_analysis.loc[df_analysis["condition"] == "A&C"]["accuracy"])
# %%

boxplot_1 = df_analysis.boxplot(column=['accuracy'], by='SOA_of_cue', grid=False, color='black', rot= 0)

# Set the title of the boxplot and remove the default title
title_boxplot_1 = "Accuracy Values Grouped by 'SOA of cue'"
plt.title(title_boxplot_1)
plt.suptitle("")
boxplot_1.set_xlabel("SOA of cue")
boxplot_1.set_ylabel("Accuracy")# %%
