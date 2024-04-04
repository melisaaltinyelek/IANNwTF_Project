# Lastest Update: 04.04.2024

# (1) import libraries
import numpy as np
import pandas as pd
import itertools
from ast import literal_eval

# (2) functions
###################################################

# Some functions

###################################################

def valid_pairs(l_input_output_pairs):
    """
    Parametes
    ------------
    l_input_output_pairs: list (4D) 

    Returns
    ------------
    valid_paired_list : list (5D) 

    Example
    -----------
    l_input_output_pairs =  
        [[ ["S1_A","S1_output_A"],  ["S1_B","S1_output_B"],   ["S1_C","S1_output_C"],  ["S1_D","S1_output_D"],  ["S1_E","S1_output_E"]],
         [ ["S2_A","S2_output_A"],  ["S2_B","S2_output_B"],   ["S2_C","S2_output_C"],  ["S2_D","S2_output_D"],  ["S2_E","S2_output_E"]],
         [ ["S3_A","S2_output_A"],  ["S3_B","S3_output_B"],   ["S3_C","S3_output_C"],  ["S3_D","S3_output_D"],  ["S3_E","S3_output_E"]],
                            ...  ,                    ... ,                  ...    ,                ...     ,         ...             ,
         [["S27_A","S27_output_A"],["S27_B","S27_output_B"],["S27_C","S27_output_C"], ["S27_D","S27_output_D"],["S27_E","S27_output_E"]]]
        

    returns: (all possible combinations - without replacement, of [l_fixed,[[l_paired,l_output]])
             [ [[["S1_A", "S1_output_B"],  ["S1_B", "S1_output_B"]],
                [["S2_A", "S1_output_A"],  ["S2_B", "S2_output_B"]]],
                            ...
               [["S27_A","S1_output_A"],["S27_B", "S27_outtput_B"]]],
            
              [[["S1_A", "S1_output_A"],  ["S1_C", "S1_ouput_C"]],
               [["S2_A", "S1_output_A"],  ["S2_C", "S2_output_C"]],
                            ...
               [["S27_A","S1_output_A"],["S27_C", "S27_output_C"]]],

               ...

              ]
    """  
    valid_paired_list = list()
    # for each stimuli S1, S2, ... , S27
    for j in range(len(l_input_output_pairs)):
        # create all possible combinations (with replacement)
        combinations = list(itertools.product(l_input_output_pairs[j],l_input_output_pairs[j]))
        # remove from combinations all entries are the same (which means no switch)
        l = list(range(len(combinations)))
        for i in l:
            if combinations[i][0] == combinations[i][1]:
                combinations.remove((combinations[i][0],combinations[i][1]))
                l = list(range(len(l)-1))
            if i == len(l):
                break
        combinations = list((list(tup) for tup in combinations))
        valid_paired_list = valid_paired_list + [combinations]
    return valid_paired_list

def df_correctly_cued(valid_paired_list, n_timesteps):
    """
    creates dataframe of training samples for CONDITIONING

    Parameters
    -----------
    valid_paired_list : list (4D)
        output of function valid_pairs
    
    n_timesteps         : int
        raises error if it is not an even number

    Returns
    -----------
    df : pd.DataFrame
        columns : {input, prev_task, curr_task, curr_output, cue_position}

    """

    # check whether n_timesteps is even integer
    if n_timesteps%2 != 0:
        raise "n_timesteps is not an even number!"
    
    df = pd.DataFrame()
    possible_cue_positions = range(int(n_timesteps/2))

    # for each stimuli S1, S2, ... , S27
    for j in range(len(valid_paired_list)):

        #for each pair in stimuli SN
        for i in range(len(valid_paired_list[j])):
            # create temporary dataframe 
            # with columns {input, prev_task, curr_task, prev_output, curr_output, cue_position}
            temp_df = pd.DataFrame()

            input = list()
            cue = [1]
            nocue = [0]
            for n in range(len(possible_cue_positions)):
                prev_stimuli_not_cued = [nocue + valid_paired_list[j][i][0][0]] * n
                prev_stimuli_cued = [cue + valid_paired_list[j][i][0][0]] * 1
                curr_stimuli_never_cued =  [nocue + valid_paired_list[j][i][1][0]] * (n_timesteps - (n + 1))
                helper_list = prev_stimuli_not_cued + prev_stimuli_cued + curr_stimuli_never_cued
                input = input + [helper_list]
            prev_task = [ valid_paired_list[j][i][0][0][-5:] ] * len(possible_cue_positions)
            curr_task = [ valid_paired_list[j][i][1][0][-5:] ] * len(possible_cue_positions)
            # prev_output = valid_paired_list[j][i][0][1] * len(possible_cue_positions)
            curr_output = [ valid_paired_list[j][i][1][1] ] * len(possible_cue_positions)
            cue_position = possible_cue_positions
            temp_df.insert(0, "input", input, True)
            temp_df.insert(1, "prev_task", prev_task, True)
            temp_df.insert(2, "curr_task", curr_task, True)
            # temp_df.insert(5, "prev_output", prev_output, True)
            temp_df.insert(3, "curr_output", curr_output, True)
            temp_df.insert(4, "cue_position", cue_position, True)
            
            # concatenate temp_df to existing df
            df = pd.concat([df, temp_df])
    return df

def convert_dataframe(df):
    """ convert problem specific dataframe 
    (change string entries back to lists and remove task_letter column)"""

    # convert string entries back to lists
    stimulus_input = [literal_eval(x) for x in df["stimulus_input"].tolist()]
    task_input = [literal_eval(x) for x in df["task_input"].tolist()]
    output = [literal_eval(x) for x in df["output"].tolist()]
    df = df.drop('output', axis=1)
    df.insert(0, "output", output, True)
    df = df.drop('task_input', axis=1)
    df.insert(0, "task_input", task_input, True)
    df = df.drop('stimulus_input', axis=1)
    df.insert(0, "stimulus_input", stimulus_input, True)

    #remove task_letter column
    df = df.drop('task_letter',axis = 1)

    return df

def prepare_list_for_valid_pairs(df):
    """
    receives problem specific dataframe and
    returns list to pass to function "valid_pairs"

    Parameters
    -----------
    df : pd.DataFrame
        columns: cue, stimulus_input, task_input, output

    Returns
    -----------
    l_input_output_pairs : list of shape (4D)
    """
    s = np.unique(df["stimulus_input"])
    #print(s)
    l_input_output_pairs = list()
    # for every stimulus S1, S2, S3,..., S27
    for j in range(len(s)):
        #print(f"s[j]: {s[j]}")
        df_temp = df.loc[(df["stimulus_input"] == s[j])].reset_index()
        h = len(df_temp)
        df_temp = convert_dataframe(df_temp)
        #print(f"h: {h}")
        l_helper = list()
        # for each task (per stimulus)
        for i in range(h):
            # concatenate task and respective output to shape
            # [[stimulus_input,task_input], [output]]
            l = [df_temp.loc[i]["stimulus_input"] + df_temp.loc[i]["task_input"]] + [df_temp.loc[i]["output"]]
            #print(l)
            l_helper = l_helper + [l]
        #print(l_helper)
        l_input_output_pairs = l_input_output_pairs + [l_helper]
    return l_input_output_pairs

# (3) create dataset
###################################################
# create dataset

###################################################
TaskPatterns = pd.DataFrame(np.array(list(itertools.product([[1,0,0], [0,1,0], [0,0,1]], repeat=3))
).flatten().reshape(27,9))

OutputPattern_TaskA = pd.concat([TaskPatterns.iloc[: , :3] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int))], axis = 1) 
OutputPattern_TaskB = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int)),TaskPatterns.iloc[: , 3:6] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int))], axis = 1) 
OutputPattern_TaskC = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int)),TaskPatterns.iloc[: , 6:9]], axis = 1) 
OutputPattern_TaskD = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int)),TaskPatterns.iloc[: , :3] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int))], axis = 1) 
OutputPattern_TaskE = pd.concat([TaskPatterns.iloc[: , 3:6] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int))], axis = 1) 

data =  {"input" : TaskPatterns.values.tolist(),
         "[1,0,0,0,0]" : OutputPattern_TaskA.values.tolist(), # outputA
         "[0,1,0,0,0]" : OutputPattern_TaskB.values.tolist(), # outputB
         "[0,0,1,0,0]" : OutputPattern_TaskC.values.tolist(), # outputC
         "[0,0,0,1,0]" : OutputPattern_TaskD.values.tolist(), # outputD
         "[0,0,0,0,1]" : OutputPattern_TaskE.values.tolist()} # outputE
df = pd.DataFrame(data)
df = pd.melt(df, id_vars= ("input"))
df.rename(columns={'input': 'stimulus_input', 'variable': 'task_input', 'value': 'output'}, inplace=True)
df.insert(3, "task_letter", ["A"] * 27 + ["B"] * 27 + ["C"] * 27 + ["D"] * 27 + ["E"] * 27)
df.reset_index(drop=True, inplace=True)

# store data
#df.to_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv", index = False)

# check for dublicate rows: 
df.astype('string')[df.astype('string').duplicated()]
df = pd.read_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv")

# (4) create training set
###################################################
# create training set for CONDITIONING the network
#       Conditioning on (1) all tasks (A,B,C,D,E) to learn dependencies bewteen A&B (vs. A&C)
#                       (2) cue indicates a switch (to later implement PRP paradigma -> the network should learn to expect a switch when cue = 1)

# examples: TaskCue, TaskCue, ... TaskCue (total: 20 times) !! Stimuli remains the same
#           A0,A0,A0,A1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           D0,D0,D0,D0,D0,D1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           D0,D0,D0,D0,D0,D0,D0,D0,D0,D1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           B0,B0,B0,B0,B0,B0,B1,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
##################################################

# create training samples for conditioning
def create_training_samples(df, n_timesteps):
    """
    returns a dataframe of training samples for CONDITIONING
    
    Parameters:
    --------
    df : pd.DataFrame
    n_timesteps : int
        results in error if it is not an even integer

    Returns
    --------
    df_step3 : pd.DataFrame
    """
    step_1 = prepare_list_for_valid_pairs(df)
    step_2 = valid_pairs(step_1)
    df_step3 = df_correctly_cued(step_2, n_timesteps)

    return df_step3
n_time = 20
df_training_to_save = create_training_samples(df = df, n_timesteps= n_time)

# save df
#df_training_to_save.to_csv("df_training_samples_for_conditioning.csv", index = False)

# convert columsn prev_task and curr_task back to strings A,B,C,D and E
#df_training_to_save = pd.read_csv("df_training_samples_for_conditioning.csv")

df_training_to_save.loc[df_training_to_save["prev_task"] == "[1, 0, 0, 0, 0]", "prev_task"] = "A"
df_training_to_save.loc[df_training_to_save["prev_task"] == "[0, 1, 0, 0, 0]", "prev_task"] = "B"
df_training_to_save.loc[df_training_to_save["prev_task"] == "[0, 0, 1, 0, 0]", "prev_task"] = "C"
df_training_to_save.loc[df_training_to_save["prev_task"] == "[0, 0, 0, 1, 0]", "prev_task"] = "D"
df_training_to_save.loc[df_training_to_save["prev_task"] == "[0, 0, 0, 0, 1]", "prev_task"] = "E"
df_training_to_save.loc[df_training_to_save["curr_task"] == "[1, 0, 0, 0, 0]", "curr_task"] = "A"
df_training_to_save.loc[df_training_to_save["curr_task"] == "[0, 1, 0, 0, 0]", "curr_task"] = "B"
df_training_to_save.loc[df_training_to_save["curr_task"] == "[0, 0, 1, 0, 0]", "curr_task"] = "C"
df_training_to_save.loc[df_training_to_save["curr_task"] == "[0, 0, 0, 1, 0]", "curr_task"] = "D"
df_training_to_save.loc[df_training_to_save["curr_task"] == "[0, 0, 0, 0, 1]", "curr_task"] = "E"

# overwrite
df_training_to_save.to_csv("df_training_samples_for_conditioning.csv", index = False)

# (5) create validation set
###################################################
# create validation set for EVALUATING the network (fixed weights!!!)
#       Conditioning on (1) all tasks (A,B,C,D,E) to learn dependencies bewteen B&A (vs. C&A)
#                       (2) cue wrongly indicates a switch (before it actually happens)(to later implement PRP paradigma -> the network should expect a switch when cue = 1)

# examples: TaskCue, TaskCue, ... TaskCue (total: 20 times)
#           B0,B0,B0,B0,B0,B0,B0,B0,B0,B1,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#           B0,B0,B0,B0,B0,B0,B0,B0,B0,B1,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#           B0,B0,B0,B0,B0,B0,B0,B1,B0,B0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#               ...
#           B1,B0,B0,B0,B0,B0,B0,B0,B0,01,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0

#       VERSUS
#           C0,C0,C0,C0,C0,C0,C0,C0,C0,C1,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#           C0,C0,C0,C0,C0,C0,C0,C0,C1,C0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#           C0,C0,C0,C0,C0,C0,C0,C1,C0,C0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
#               ...
#           C1,C0,C0,C0,C0,C0,C0,C0,C0,C0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
###################################################
def df_incorrectly_cued(valid_paired_list, n_timesteps):
    """
    creates dataframe of validation samples for EVALUATION

    Parameters
    -----------
    valid_paired_list : list (4D)
        output of function valid_pairs
    
    n_timesteps         : int
        raises error if it is not an even number

    Returns
    -----------
    df : pd.DataFrame
        columns : {input, prev_task, curr_task, curr_output, cue_position}

    """

    # check whether n_timesteps is even integer
    if n_timesteps%2 != 0:
        raise "n_timesteps is not an even number!"
    
    df = pd.DataFrame()
    possible_cue_positions = range(int(n_timesteps/2))

    # for each stimuli S1, S2, ... , S27
    for j in range(len(valid_paired_list)):

        #for each pair in stimuli SN
        for i in range(len(valid_paired_list[j])):
            # create temporary dataframe 
            # with columns {input, prev_task, curr_task, prev_output, curr_output, cue_position}
            temp_df = pd.DataFrame()

            input = list()
            cue = [1]
            nocue = [0]
            for n in range(len(possible_cue_positions)):
                prev1_stimuli_not_cued = [nocue + valid_paired_list[j][i][0][0]] * n
                prev_stimuli_cued = [cue + valid_paired_list[j][i][0][0]] * 1
                prev2_stimuli_not_cued = [nocue + valid_paired_list[j][i][0][0]] * (int(n_timesteps/2) - (n +  1))
                curr_stimuli_never_cued =  [nocue + valid_paired_list[j][i][1][0]] * (int(n_timesteps/2))
                helper_list = prev1_stimuli_not_cued + prev_stimuli_cued + prev2_stimuli_not_cued + curr_stimuli_never_cued
                input = input + [helper_list]
            prev_task = [ valid_paired_list[j][i][0][0][-5:] ] * len(possible_cue_positions)
            curr_task = [ valid_paired_list[j][i][1][0][-5:] ] * len(possible_cue_positions)
            # prev_output = valid_paired_list[j][i][0][1] * len(possible_cue_positions)
            curr_output = [ valid_paired_list[j][i][1][1] ] * len(possible_cue_positions)
            cue_position = possible_cue_positions
            temp_df.insert(0, "input", input, True)
            temp_df.insert(1, "prev_task", prev_task, True)
            temp_df.insert(2, "curr_task", curr_task, True)
            # temp_df.insert(5, "prev_output", prev_output, True)
            temp_df.insert(3, "curr_output", curr_output, True)
            temp_df.insert(4, "cue_position", cue_position, True)
            
            # concatenate temp_df to existing df
            df = pd.concat([df, temp_df])
    return df

# create validation samples for evaluation
def create_validation_samples(df, n_timesteps):
    """
    returns a dataframe of validation samples for EVALUATION

    Parameters:
    --------
    df : pd.DataFrame
    n_timesteps : int
        results in error if it is not an even integer

    Returns
    --------
    df_step3 : pd.DataFrame
    """
    step_1 = prepare_list_for_valid_pairs(df)
    step_2 = valid_pairs(step_1)
    df_step3 = df_incorrectly_cued(step_2, n_timesteps)
    return df_step3

# for validation we evaluate taks A,B,C only
n_time = 20
df_validation = df.loc[(df["task_letter"] == "A") | (df["task_letter"] == "B") | (df["task_letter"] == "C")]
df_val_to_save = create_validation_samples(df = df_validation, n_timesteps= n_time)

#df_val_to_save.to_csv("df_validation_samples_for_evaluation.csv", index = False)

# convert columns prev_task and curr_task back to strings A,B,C,D and E
df_val_to_save = pd.read_csv("df_validation_samples_for_evaluation.csv")

df_val_to_save.loc[df_val_to_save["prev_task"] == "[1, 0, 0, 0, 0]", "prev_task"] = "A"
df_val_to_save.loc[df_val_to_save["prev_task"] == "[0, 1, 0, 0, 0]", "prev_task"] = "B"
df_val_to_save.loc[df_val_to_save["prev_task"] == "[0, 0, 1, 0, 0]", "prev_task"] = "C"
df_val_to_save.loc[df_val_to_save["curr_task"] == "[1, 0, 0, 0, 0]", "curr_task"] = "A"
df_val_to_save.loc[df_val_to_save["curr_task"] == "[0, 1, 0, 0, 0]", "curr_task"] = "B"
df_val_to_save.loc[df_val_to_save["curr_task"] == "[0, 0, 1, 0, 0]", "curr_task"] = "C"

df_val_to_save = df_val_to_save.loc[df_val_to_save["curr_task"] == "A"]

# overwrite
df_val_to_save.to_csv("df_validation_samples_for_evaluation.csv", index = False)
