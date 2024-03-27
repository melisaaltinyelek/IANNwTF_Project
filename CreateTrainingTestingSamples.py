#%%
import numpy as np
import pandas as pd
import itertools
from ast import literal_eval
# %%
# create dataset
TaskPatterns = pd.DataFrame(np.array(list(itertools.product([[1,0,0], [0,1,0], [0,0,1]], repeat=3))
).flatten().reshape(27,9))

OutputPattern_TaskA = pd.concat([TaskPatterns.iloc[: , :3] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int))], axis = 1) 
OutputPattern_TaskB = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int)),TaskPatterns.iloc[: , 3:6] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int))], axis = 1) 
OutputPattern_TaskC = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int)),TaskPatterns.iloc[: , 6:9]], axis = 1) 
OutputPattern_TaskD = pd.concat([pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int)),TaskPatterns.iloc[: , :3] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 3)).astype(int))], axis = 1) 
OutputPattern_TaskE = pd.concat([TaskPatterns.iloc[: , 3:6] , pd.DataFrame(np.zeros(shape=(len(TaskPatterns), 6)).astype(int))], axis = 1) 

data =  {"input" : TaskPatterns.values.tolist(),
         "[1,0,0,0,0]" : OutputPattern_TaskA.values.tolist(), # outputA
         "[0,1,0,0,0]" : OutputPattern_TaskB.values.tolist(), # outputA
         "[0,0,1,0,0]" : OutputPattern_TaskC.values.tolist(), # outputC
         "[0,0,0,1,0]" : OutputPattern_TaskD.values.tolist(), # outputD
         "[0,0,0,0,1]" : OutputPattern_TaskE.values.tolist()} # outputE
df = pd.DataFrame(data)
df = pd.melt(df, id_vars= ("input"))
df.rename(columns={'input': 'stimulus_input', 'variable': 'task_input', 'value': 'output'}, inplace=True)
t0  = [0] * len(df)
t1 = [1] * len(df)
df = pd.concat([df,df])
df.insert(0, "cue", t0 + t1, True)
df.reset_index(drop=True, inplace=True)
#%%
# convert to correct dataformat
# task_input = [literal_eval(x) for x in df["task_input"].tolist()]
# df = df.drop('task_input', axis=1)
# df.insert(1, "task_input", task_input, True)

# %%
# store data
# df.to_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv", index = False)
# %%
# check for dublicate rows: 
df.astype('string')[df.astype('string').duplicated()]


# %%
#################
# create training set for CONDITIONING the network
#       Conditioning on (1) all tasks (A,B,C,D,E) to learn dependencies bewteen A&B (vs. A&C)
#                       (2) cue indicates a switch (to later implement PRP paradigma -> the network should expect a switch when cue = 1)

# examples: TaskCue, TaskCue, ... TaskCue (total: 20 times)
#           A0,A0,A0,A1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           D0,D0,D0,D0,D0,D1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           D0,D0,D0,D0,D0,D0,D0,D0,D0,D1,B0,B0,B0,B0,B0,B0,B0,B0,B0,B0
#           B0,B0,B0,B0,B0,B0,B1,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0,A0
################

#%%
#read df
#df_original = pd.read_csv("ALLInputOutputSamples_TasksABCDE_withcues0.csv")

#%%
def flatten_list(list_to_be_flattened):
    """ Takes a nested input list and concatenates them into a single list.
    
    Parameters
    ----------
    list_to_be_flattened : list

    Returns
    ----------
    flattened_list : list
    """

    flattened_list = [num for elem in list_to_be_flattened for num in elem]
    return flattened_list

def valid_pairs(l_fixed, l_paired):
    """
    Parametes
    ------------
    l_fixed : list (2D)
              is the fixed task list
    l_paired : list (2D)
              list to be paired with l_fixed
    
    Returns
    ------------
    valid_paired_list : list (3D)    

    Example:
    -----------
    l_fixed = [["A1_1","A1_2"],["A2_1","A2_2"]], l_paired = [["B1_2","B1_2","B1_3"],["C1_2","C1_2","C1_3"],["D1_2","D1_2","D1_3"]]
    returns: 
            [[['A1_1', 'A1_2'], ['B1_2', 'B1_2', 'B1_3']],
            [['A1_1', 'A1_2'], ['C1_2', 'C1_2', 'C1_3']],
            [['A1_1', 'A1_2'], ['D1_2', 'D1_2', 'D1_3']],
            [['A2_1', 'A2_2'], ['B1_2', 'B1_2', 'B1_3']],
            [['A2_1', 'A2_2'], ['C1_2', 'C1_2', 'C1_3']],
            [['A2_1', 'A2_2'], ['D1_2', 'D1_2', 'D1_3']]]
    """
    l = []
    l_fixed_temp = []
    for i in range(len(l_fixed)):
        r = [l_fixed[i]] * len(l_paired)
        l_fixed_temp.append(r)
    l_temp_fixed = flatten_list(l_fixed_temp)
    l_temp_paired = l_paired * len(l_fixed)
    for i in range(len(l_temp_fixed)):
        r = [l_temp_fixed[i],l_temp_paired[i]]
        l.append(r)
    valid_paired_list = l
    return valid_paired_list

def convert_dataframe(df):
    """ convert problem specific dataframe 
    (change string entries back to lists and remove output column)"""
    stimulus_input = [literal_eval(x) for x in df["stimulus_input"].tolist()]
    task_input = [literal_eval(x) for x in df["task_input"].tolist()]
    cue = [[x] for x in df["cue"]]

    df = df.drop('output', axis=1)
    df = df.drop("index", axis = 1)
    df = df.drop('task_input', axis=1)
    df.insert(0, "task_input", task_input, True)
    df = df.drop('stimulus_input', axis=1)
    df.insert(0, "stimulus_input", stimulus_input, True)
    df = df.drop('cue', axis=1)
    #df.insert(0, "cue", cue, True)
    return df

#%%
# for each task
tasksABCDE = ['[1, 0, 0, 0, 0]','[0, 1, 0, 0, 0]','[0, 0, 1, 0, 0]','[0, 0, 0, 1, 0]','[0, 0, 0, 0, 1]']
tasksABCDE_letter = ["A","B","C","D","E"]
df = df_original.loc[df_original["cue"] == 0]
df_conditioning_incomplete = "NOT INSTANTIATED"
for i in range(len(tasksABCDE)):
    # for each instance in a task
    df_temp_l_fixed = df.loc[(df["task_input"] == tasksABCDE[i])].reset_index()
    l_temp_fixed = convert_dataframe(df_temp_l_fixed).values.tolist()
    l_fixed = [flatten_list(x) for x in l_temp_fixed]
    # for each possible task other than selected
    tasksABCDE_letter.remove(tasksABCDE_letter[i])
    other_tasksABCDE_letter = tasksABCDE_letter
    tasksABCDE_letter = ["A","B","C","D","E"]
    tasksABCDE.remove(tasksABCDE[i])
    other_tasksABCDE = tasksABCDE
    tasksABCDE = ['[1, 0, 0, 0, 0]','[0, 1, 0, 0, 0]','[0, 0, 1, 0, 0]','[0, 0, 0, 1, 0]','[0, 0, 0, 0, 1]']
    # print(other_tasksABCDE_letter)
    helper = len(other_tasksABCDE_letter)
    for n in range(helper):
        df_temp_l_paired = df.loc[(df["task_input"] == other_tasksABCDE[n])].reset_index()
        l_temp_paired = convert_dataframe(df_temp_l_paired).values.tolist()
        l_paired = [flatten_list(x) for x in l_temp_paired]
        valid_pairs_two = valid_pairs(l_fixed=l_fixed, l_paired=l_paired)
        df_pairs = pd.DataFrame(valid_pairs_two).rename(columns={0:"prev_input", 1: "curr_input"})
        df_pairs.insert(2, "prev_task", [tasksABCDE_letter[i]]*len(df_pairs), True)
        df_pairs.insert(3, "curr_task", [other_tasksABCDE_letter[n]]*len(df_pairs), True)
        if isinstance(df_conditioning_incomplete,str):
            df_conditioning_incomplete = df_pairs
            #print(df_conditioning_incomplete)
        else:
            df_conditioning_incomplete = pd.concat([df_conditioning_incomplete,df_pairs])
            #print(df_pairs)
df_conditioning_incomplete = df_conditioning_incomplete.reset_index()

#%%
def insert_cue_to_pairs(df_conditioning_incomplete, n_timesteps):
    """
    extends the problem specific dataframe with the information when a correct cue occurs

    Parameters
    -----------
    df_conditioning_incomplete: pd.DataFrame
                columns: {prev_input, curr_input, prev_task, curr_task}
                prev_input: lists (1D) of shape (14,)
                curr_input: list (1D) of shape (14,)
                prev_task  : str ("A","B","C","D") of prev_input
                curr_task  : str ("A","B","C","D") of curr_input
    n_timesteps: int 
                raises error if it is not an even integer
            
    Returns
    -----------
    df_conditioning : pd.DataFrame
                columns: {input, prev_task, curr_task}
                input       : list (2D) of shape (n_timesteps,15)
                prev_task  : str ("A","B","C","D") of prev_input
                curr_task  : str ("A","B","C","D") of curr_input
    """
    n_timesteps = int(n_timesteps)
    if n_timesteps%2 != 0:
        raise "Number of timesteps is uneven!"
    if isinstance(df_conditioning_incomplete,str):
        raise "Dataframe given is not instantiated!"
    
    df_conditioning = pd.DataFrame()
    cue_position = list(range(int(n_timesteps/2)))
    # iterate each pair of tasks in df_conditioning_incomplete
    for i in range(len(df_conditioning_incomplete)):
        # for each cue_position
        temp_helper_sample = list()
        for n in range(len(cue_position)):  
            # create (n_timesteps)*(cue_position) many new rows for each pair
            temp_sample = list()
            cued_list = [1]
            notcued_list1 = [0]
            notcued_list2 = [0]
            task_1_notcued = [(notcued_list1 + df_conditioning_incomplete["prev_input"][i])] * (cue_position[n])
            task_1_cued = [(cued_list + df_conditioning_incomplete["prev_input"][i])] * 1
            task_2_nevercued = [(notcued_list2 + df_conditioning_incomplete["curr_input"][i])] * (n_timesteps - (cue_position[n] + 1))
            temp_sample = task_1_notcued + task_1_cued + task_2_nevercued
            temp_helper_sample.append(temp_sample)

        df_temp= pd.DataFrame()
        df_temp.insert(0, "input", [temp_helper_sample] ,True)
        df_temp.insert(1,"prev_task", [df_conditioning_incomplete["prev_task"][i]])
        df_temp.insert(2,"curr_task", [df_conditioning_incomplete["curr_task"][i]])
        df_conditioning = pd.concat([df_conditioning, df_temp])

        if i%1000 == 0:
            print(i/len(df_conditioning_incomplete))
    df_conditioning = df_conditioning.reset_index()
    df_conditioning = df_conditioning.drop("index", axis = 1)
    return df_conditioning


df_conditioning_to_save = insert_cue_to_pairs(df_conditioning_incomplete ,n_timesteps=20)

#%%
# store the data as .csv file
df_conditioning_to_save.to_csv("df_training_samples_for_conditioning.csv", index = False)
#%%
#################
# create training set for TESTING the network
#       Conditioning on (1) all tasks (A,B,C,D,E) to learn dependencies bewteen A&B (vs. A&C)
#                       (2) cue wrongly indicates a switch (before it actually happens)(to later implement PRP paradigma -> the network should expect a switch when cue = 1)

# examples: TaskCue, TaskCue, ... TaskCue (total: 20 times)
#           A0,A0,A0,A1,B0,B0,B0,B0,B0,B0
#           D0,D0,D0,D0,D0,D1,B0,B0,B0,B0
#           D0,D0,D0,D1,B0,B0,B0,B0,B0,B0

################
