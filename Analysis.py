#%%
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import matplotlib.pyplot as plt
#%%

# Read data
df_AB = pd.read_csv("df_AB_all_accuracies_run1.csv")
df_AC = pd.read_csv("df_AC_all_accuracies_run1.csv")
df_loss_accuracies = pd.read_csv("loss_df_run1.csv")

# Prepare data for analysis
df_AB.insert(0, "condition", ["A&B"] * len(df_AB))
df_AC.insert(0, "condition", ["A&C"] * len(df_AC))
df_concat = pd.concat([df_AB, df_AC], axis = 0)
df_concat.insert(0, "accuracy", list(df_loss_accuracies["accuracy"]) * 2)

# Filter rows where accuracy is greater or equal than 0.9 = 90% for analysis
df_analysis = df_concat.loc[(df_concat["accuracy"]  >=  0.9)]
df_analysis = df_analysis.loc[:, ~df_analysis.columns.str.startswith('Un')]
df_analysis = df_analysis.loc[:, ~df_analysis.columns.str.startswith('in')]
df_analysis = pd.melt(df_analysis, id_vars='condition', value_vars=["val_acc_cuepos0" ,
"val_acc_cuepos1",	"val_acc_cuepos2",	"val_acc_cuepos3",	"val_acc_cuepos4",	"val_acc_cuepos5",	"val_acc_cuepos6",	"val_acc_cuepos7",	"val_acc_cuepos8",	"val_acc_cuepos9"])
df_analysis = df_analysis.rename(columns={"variable" : "cue_position", "value" : "accuracy"})

#%%

# (Two-Way ANOVA (Typ III)) .. levene test ist missing
model = ols('accuracy ~ C(condition) + C(cue_position) + C(condition):C(cue_position)', data=df_analysis).fit()
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

# Create boxplot based on condition and accuracy
boxplot_1 = df_analysis.boxplot(column=['accuracy'], by='condition', grid=False, color='black')

# Set the title of the boxplot and remove the default title
title_boxplot_1 = "Accuracy Values Grouped by Condition"
plt.title(title_boxplot_1)
plt.suptitle("")
boxplot_1.set_xlabel("Condition")
boxplot_1.set_ylabel("Accuracy")

# %%

# Create boxplot based on the cue position and accuracy
boxplot_2 = df_analysis.boxplot(column=['accuracy'], by='cue_position', grid=False, color='black', rot= 90)

#Set the title of the boxplot and remove the default title
title_boxplot_2 = "Accuracy Values Grouped by Cue Position"
plt.title(title_boxplot_2)
plt.suptitle("")
boxplot_2.set_xlabel("Cue position")
boxplot_2.set_ylabel("Accuracy")
# %%
