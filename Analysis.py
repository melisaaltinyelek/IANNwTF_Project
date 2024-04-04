#%%
# import necessary libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.formula.api import ols
import scipy.stats as stats
import matplotlib.pyplot as plt

# %%
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

# for analysis filter rows where loss is smaller or equal than 0.01
df_analysis = df_concat.loc[(df_concat["loss"]  <=  0.01)]
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


# (Two-Way ANOVA (Typ III)) .. levene test ist missing
model = ols('accuracy ~ C(condition) + C(SOA_of_cue) + C(condition):C(SOA_of_cue)', data=df_analysis).fit()
sm.stats.anova_lm(model, typ=3)


# Two-sample t-test (Welch)
stats.ttest_ind(a=list(df_analysis.loc[df_analysis["condition"] == "A&B"]["accuracy"]), b=list(df_analysis.loc[df_analysis["condition"] == "A&C"]["accuracy"]), equal_var= False)


# create boxplot
boxplot_0 = df_analysis.boxplot(column=['accuracy'], by='condition', grid=False, color='black')
# Set the title of the boxplot and remove the default title
title_boxplot_1 = "Accuracy Values Grouped by Condition"
plt.title(title_boxplot_1)
plt.suptitle("")
boxplot_0.set_xlabel("Condition")
boxplot_0.set_ylabel("Accuracy")# %%


boxplot_1 = df_analysis.boxplot(column=['accuracy'], by='SOA_of_cue', grid=False, color='black', rot= 0)

# Set the title of the boxplot and remove the default title
title_boxplot_1 = "Accuracy Values Grouped by 'SOA of cue'"
plt.title(title_boxplot_1)
plt.suptitle("")
boxplot_1.set_xlabel("SOA of cue")
boxplot_1.set_ylabel("Accuracy")# %%

# %%
# prepare data for plotting
def X_Y_of_create_df_graph(df_concat, cue_pos):
    """
    Returns X-values, Y-values and standard deviation for plotting
    """

    df_graphs = df_concat.loc[:, ~df_concat.columns.str.startswith('in')]
    df_graphs= df_graphs.rename(columns={"val_acc_cuepos0" : "0" ,
                                        "val_acc_cuepos1" : "1",	
                                        "val_acc_cuepos2" : "2",	
                                        "val_acc_cuepos3" : "3",	
                                        "val_acc_cuepos4" : "4",	
                                        "val_acc_cuepos5" : "5",	
                                        "val_acc_cuepos6" : "6",	
                                        "val_acc_cuepos7" : "7",	
                                        "val_acc_cuepos8" : "8",	
                                        "val_acc_cuepos9" : "9",
                                        "Unnamed: 0"      : "epoch"})
    df_graphs =  pd.melt(df_graphs, id_vars=['epoch'], value_vars=["0","1","2","3","4","5","6","7","8","9"])
    df_graphs = df_graphs.loc[df_graphs["variable"] == str(cue_pos)]
    df_graphs = df_graphs.drop("variable", axis = 1)
    print(df_graphs.columns)
    df_g = df_graphs.groupby(by = "epoch").mean()
    df_err = df_graphs.groupby(by = "epoch").std()
    X = list(range(len(list(df_g["value"]))))
    Y = list(df_g["value"]) 
    return X, Y, df_err["value"]

# plot Mean validation set accuracies over epochs 2 to 9 across both conditions A&B and A&C 
df_concat_a = df_concat.loc[(df_concat["loss"]  <=  0.01)]
X0, Y0, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=0)
X1, Y1, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=1)
X2, Y2, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=2)
X3, Y3, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=3)
X4, Y4, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=4)
X5, Y5, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=5)
X6, Y6, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=6)
X7, Y7, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=7)
X8, Y8, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=8)
X9, Y9, std_values = X_Y_of_create_df_graph(df_concat_a, cue_pos=9)
plt.plot([x+2 for x in X0], Y0, label = "0")
plt.plot([x+2 for x in X1], Y1, label = "1")
plt.plot([x+2 for x in X2], Y2, label = "2")
plt.plot([x+2 for x in X3], Y3, label = "3")
plt.plot([x+2 for x in X4], Y4, label = "4")
plt.plot([x+2 for x in X5], Y5, label = "5")
plt.plot([x+2 for x in X6], Y6, label = "6")
plt.plot([x+2 for x in X7], Y7, label = "7")
plt.plot([x+2 for x in X8], Y8, label = "8")
plt.plot([x+2 for x in X9], Y9, label = "9")
plt.legend().set_title("SOA of cue")
plt.title("Average validation set accuracies per epoch accross both conditions A&B and A&C")
plt.xlabel("epoch")
plt.ylabel("accuracy score")


# mean number of epochs
np.mean(df_concat["Unnamed: 0"] + 1)

# std of number of epochs
np.std(df_concat["Unnamed: 0"] + 1)

# plot accuracies 
Y = list(df_concat.loc[:,["Unnamed: 0","accuracy"]].groupby("Unnamed: 0").mean()["accuracy"])
X = list(range(len(Y)))
plt.plot(X,Y)
std_values = list(df_concat.loc[:,["Unnamed: 0","accuracy"]].groupby("Unnamed: 0").std()["accuracy"])
plt.errorbar(X,Y,
             yerr = std_values,
             fmt = 'o',
             ecolor = "orange")
plt.title("Average training set accuracy scores")
plt.xlabel("epoch")
plt.ylabel("accuracy score")

# plot loss (MSE)
Y = list(df_concat.loc[:,["Unnamed: 0","loss"]].groupby("Unnamed: 0").mean()["loss"])
X = list(range(len(Y)))
plt.plot(X,Y)
std_values = list(df_concat.loc[:,["Unnamed: 0","loss"]].groupby("Unnamed: 0").std()["loss"])
plt.errorbar(X,Y,
             yerr = std_values,
             fmt = 'o',
             color = "red",
             ecolor = "red")
plt.title("average MSE loss across 30 networks")
plt.xlabel("epoch")
plt.ylabel("MSE loss")

