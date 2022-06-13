from google.colab import drive
drive.mount('/content/drive/')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import os
from os import path
from sklearn.cluster import KMeans

#@title ProgSnap Code
progsnap = "" #@param {type:"string"}
import pandas as pd
import os
from os import path


class PS2:
    """ A class holding constants used to get columns of a PS2 dataset
    """

    Order = 'Order'
    SubjectID = 'SubjectID'
    ToolInstances = 'ToolInstances'
    ServerTimestamp = 'ServerTimestamp'
    ServerTimezone = 'ServerTimezone'
    CourseID = 'CourseID'
    CourseSectionID = 'CourseSectionID'
    AssignmentID = 'AssignmentID'
    ProblemID = 'ProblemID'
    Attempt = 'Attempt'
    CodeStateID = 'CodeStateID'
    IsEventOrderingConsistent = 'IsEventOrderingConsistent'
    EventType = 'EventType'
    Score = 'Score'
    CompileResult = 'CompileResult'
    CompileMessageType = 'CompileMessageType'
    CompileMessageData = 'CompileMessageData'
    EventID = 'EventID'
    ParentEventID = 'ParentEventID'
    SourceLocation = 'SourceLocation'
    Code = 'Code'

    Version = 'Version'
    IsEventOrderingConsistent = 'IsEventOrderingConsistent'
    EventOrderScope = 'EventOrderScope'
    EventOrderScopeColumns = 'EventOrderScopeColumns'
    CodeStateRepresentation = 'CodeStateRepresentation'


class ProgSnap2Dataset:

    MAIN_TABLE_FILE = 'MainTable.csv'
    METADATA_TABLE_FILE = 'DatasetMetadata.csv'
    LINK_TABLE_DIR = 'LinkTables'
    CODE_STATES_DIR = 'CodeStates'
    CODE_STATES_TABLE_FILE = os.path.join(CODE_STATES_DIR, 'CodeStates.csv')

    def __init__(self, directory):
        self.directory = directory
        self.main_table = None
        self.metadata_table = None
        self.code_states_table = None

    def path(self, local_path):
        return path.join(self.directory, local_path)

    def get_main_table(self):
        """ Returns a Pandas DataFrame with the main event table for this dataset
        """
        if self.main_table is None:
            self.main_table = pd.read_csv(self.path(ProgSnap2Dataset.MAIN_TABLE_FILE))
            if self.get_metadata_property(PS2.IsEventOrderingConsistent):
                order_scope = self.get_metadata_property(PS2.EventOrderScope)
                if order_scope == 'Global':
                    # If the table is globally ordered, sort it
                    self.main_table.sort_values(by=[PS2.Order], inplace=True)
                elif order_scope == 'Restricted':
                    # If restricted ordered, sort first by grouping columns, then by order
                    order_columns = self.get_metadata_property(PS2.EventOrderScopeColumns)
                    if order_columns is None or len(order_columns) == 0:
                        raise Exception('EventOrderScope is restricted by no EventOrderScopeColumns given')
                    columns = order_columns.split(';')
                    columns.append('Order')
                    # The result is that _within_ these groups, events are ordered
                    self.main_table.sort_values(by=columns, inplace=True)
        return self.main_table.copy()

    def set_main_table(self, main_table):
        """ Overwrites the main table loaded from the file with the provided table.
        This this table will be used for future operations, including copying the dataset.
        """
        self.main_table = main_table.copy()

    def get_code_states_table(self):
        """ Returns a Pandas DataFrame with the code states table form this dataset
        """
        if self.code_states_table is None:
            self.code_states_table = pd.read_csv(self.path(ProgSnap2Dataset.CODE_STATES_TABLE_FILE))
        return self.code_states_table.copy()

    def get_metadata_property(self, property):
        """ Returns the value of a given metadata property in the metadata table
        """
        if self.metadata_table is None:
            self.metadata_table = pd.read_csv(self.path(ProgSnap2Dataset.METADATA_TABLE_FILE))

        values = self.metadata_table[self.metadata_table['Property'] == property]['Value']
        if len(values) == 1:
            return values.iloc[0]
        if len(values) > 1:
            raise Exception('Multiple values for property: ' + property)

        # Default return values as of V6
        if property == PS2.IsEventOrderingConsistent:
            return False
        if property == PS2.EventOrderScope:
            return 'None'
        if property == PS2.EventOrderScopeColumns:
            return ''

        return None

    def __link_table_path(self):
        return self.path(ProgSnap2Dataset.LINK_TABLE_DIR)

    def list_link_tables(self):
        """ Returns a list of the link tables in this dataset, which can be loaded with load_link_table
        """
        path = self.__link_table_path()
        dirs = os.listdir(path)
        return [f for f in dirs if os.path.isfile(os.path.join(path, f)) and f.endswith('.csv')]

    def load_link_table(self, link_table):
        """ Returns a Pandas DataFrame with the link table with the given name
        :param link_table: The link table nme or file
        """
        if not link_table.endswith('.csv'):
            link_table += '.csv'
        return pd.read_csv(path.join(self.__link_table_path(), link_table))

    def drop_main_table_column(self, column):
        self.get_main_table()
        self.main_table.drop(column, axis=1, inplace=True)

    def save_subset(self, path, main_table_filterer, copy_link_tables=True):
        os.makedirs(os.path.join(path, ProgSnap2Dataset.CODE_STATES_DIR), exist_ok=True)
        main_table = main_table_filterer(self.get_main_table())
        main_table.to_csv(os.path.join(path, ProgSnap2Dataset.MAIN_TABLE_FILE), index=False)
        code_state_ids = main_table[PS2.CodeStateID].unique()
        code_states = self.get_code_states_table()
        code_states = code_states[code_states[PS2.CodeStateID].isin(code_state_ids)]
        code_states.to_csv(os.path.join(path, ProgSnap2Dataset.CODE_STATES_DIR, 'CodeStates.csv'), index=False)
        self.metadata_table.to_csv(os.path.join(path, ProgSnap2Dataset.METADATA_TABLE_FILE), index=False)

        if not copy_link_tables:
            return

        os.makedirs(os.path.join(path, ProgSnap2Dataset.LINK_TABLE_DIR), exist_ok=True)

        def indexify(x):
            return tuple(x) if len(x) > 1 else x[0]

        for link_table_name in self.list_link_tables():
            link_table = self.load_link_table(link_table_name)
            columns = [col for col in link_table.columns if col.endswith('ID') and col in main_table.columns]
            distinct_ids = main_table.groupby(columns).apply(lambda x: True)
            # TODO: Still need to test this with multi-ID link tables
            to_keep = [indexify(list(row)) in distinct_ids for index, row in link_table[columns].iterrows()]
            filtered_link_table = link_table[to_keep]
            filtered_link_table.to_csv(os.path.join(path, ProgSnap2Dataset.LINK_TABLE_DIR, link_table_name), index=False)



    @staticmethod
    def __to_one(lst, error):
        if len(lst) == 0:
            return None
        if len(lst) > 1:
            raise Exception(error or 'Should have only one result!')
        return lst.iloc[0]

    def get_code_for_id(self, code_state_id):
        if code_state_id is None:
            return None
        code_states = self.get_code_states_table()
        code = code_states[code_states[PS2.CodeStateID] == code_state_id][PS2.Code]
        return ProgSnap2Dataset.__to_one(code, 'Multiple code states match that ID.')

    def get_code_for_event_id(self, row_id):
        events = self.get_main_table()
        code_state_ids = events[events[PS2.EventID == row_id]][PS2.CodeStateID]
        code_state_id = ProgSnap2Dataset.__to_one(code_state_ids, 'Multiple rows match that ID.')
        return self.get_code_for_id(code_state_id)

    def get_subject_ids(self):
        events = self.get_main_table()
        return events[PS2.SubjectID].unique()

    def get_problem_ids(self):
        events = self.get_main_table()
        return events[PS2.ProblemID].unique()

    def get_trace(self, subject_id, problem_id):
        events = self.get_main_table()
        rows = events[(events[PS2.SubjectID] == subject_id) & (events[PS2.ProblemID] == problem_id)]
        ids = rows[PS2.CodeStateID].unique()
        return [self.get_code_for_id(code_state_id) for code_state_id in ids]


if __name__ == '__main__':
  data = ProgSnap2Dataset('/content/drive/Shareddrives/Learning Analytics/data/Release/S19/Train/Data/')    # for code in data.get_trace('4d230b683bf9840553ae57f4acc96e81', 32):
    #     print(code)
    #     print('-------')

  data.save_subset('data/test/CopyA', lambda df: df[df[PS2.SubjectID].str.startswith('a')])

TRAIN_PATH = '/content/drive/Shareddrives/Learning Analytics/data/Release/S19/Train'

train_ps2 = ProgSnap2Dataset(os.path.join(TRAIN_PATH, 'Data'))
main_table = train_ps2.get_main_table()

#@title EQ code AND SCORE REGRESSION FEATURE

from enum import Enum
from sklearn.linear_model import LinearRegression


def get_error_type(error):
  if "cannot find symbol: variable" in error:
    return 1
  elif "';' expected" in error:
    return 2
  elif "'(' expected" in error or "expected" in error or "')' expected" in error or "'[' expected" in error or "']' expected" in error or "'{' expected" in error or "'}' expected" in error :
    return 3
  elif "missing return statement" in error:
    return 4
  elif "cannot find symbol: method" in error or "cannot find symbol" in error:
    return 5
  elif "illegal start of" in error:
    return 6
  elif "incompatible types" in error:
    return 7
  elif "<identifier> expected" in error:
    return 8
  elif "class, interface, or enum expected" in error:
    return 9
  elif "'else' without 'if'" in error:
    return 10
  elif "bad operand" in error:
    return 11
  elif "cannot be dereferenced" in error:
    return 12
  elif "incomparable types" in error:
    return 13
  elif "illegal character" in error or "illegal" in error:
    return 14
  elif "not a statement" in error:
    return 15
  elif "might not have been initialized" in error:
    return 16
  elif "unreachable statement" in error:
    return 17
  elif "no suitable method found" in error:
    return 18
  elif "reached end of file while parsing" in error:
    return 19
  elif "unclosed" in error or "literal" in error:
    return 20
  elif "is already defined" in error:
    return 21
  elif "empty statement after if" in error:
    return 22
  elif "variable declaration not allowed here" in error:
    return 23
  elif "array required" in error:
    return 24
  elif "invalid method declaration" in error:
    return 25
  elif "not applicable" in error:
    return 26
  elif "cannot be applied" in error or "cannot assign" in error:
    return 27
  elif "no suitable constructor" in error:
    return 28
  elif "cannot be referenced" in error:
    return 29
  elif "bad initializer" in error:
    return 30
  elif "does not exist" in error:
    return 31
  else:
    return -1




##This is the table were going to do the logic to make it easy to implment Jaduds algorithm

regression_score =  main_table[["SubjectID", "ServerTimestamp","ProblemID","EventType","Score","Compile.Result", "CompileMessageType", "CompileMessageData"]]
regression_score = regression_score[regression_score.EventType == "Run.Program"]

eq_algo = main_table[["SubjectID", "ServerTimestamp","ProblemID","EventType", "Compile.Result", "CompileMessageType", "CompileMessageData"]]

eq_algo["ServerTimestamp"] = pd.to_datetime(eq_algo["ServerTimestamp"], format='%Y-%m-%dT%H:%M:%S')

eq_algo[["CompileMessageData", "CompileMessageType"]] = eq_algo[["CompileMessageData", "CompileMessageType"]].shift(-1)
eq_algo = eq_algo[eq_algo.EventType == "Compile"]


eq_algo["ServerTimestamp"] = pd.to_datetime(eq_algo["ServerTimestamp"], format='%Y-%m-%dT%H:%M:%S')


def make_pairs(error_arr):
  err_pairs = list(map(list, zip(error_arr, error_arr[1:])))
  return err_pairs
  

import math
from math import nan
def get_eq_score(c_r, c_m, eq_1, eq_2):

  ## c_r is compile result
  c_r_pair = make_pairs(c_r)

  ##3 c_m is compile message
  c_m_pair = make_pairs(c_m)
  errors_for_pairs = []
  tot = 0.0
  for i in range(len(c_r_pair)):
    curr_eq = 0
    if c_r_pair[i][0] == "Error" and c_r_pair[i][1] == "Error":
      curr_eq = curr_eq + eq_1   
      if(get_error_type(c_m_pair[i][0]) == get_error_type(c_m_pair[i][1])):
        curr_eq = curr_eq + eq_2      
      tot = tot + (curr_eq / (eq_1 + eq_2))
    elif c_r_pair[i][0] == "Error" or c_r_pair[i][1] == "Error":
      curr_eq = (curr_eq + eq_1) / (eq_1 + eq_2)

  return tot/len(c_r_pair)


def get_get_time_between_events(server_time_stamps):
  time_diff_arr = []
  for i in range(len(server_time_stamps) - 1):
    time_diff = server_time_stamps[i+1] - server_time_stamps[i]
    seconds = time_diff/np.timedelta64(1, 's') 
    time_diff_arr.append(seconds)

  under20 = 0
  over120 = 0
  for j in time_diff_arr:
    if j < 20:
      under20 += 1
    elif j > 110:
      over120 +=1

  under20 = under20/len(time_diff_arr)
  over120 = over120/len(time_diff_arr)


  return [under20, over120]

  
def get_eq_late_scores(df, eq_1, eq_2):
  subject_ids = [id[0] for id in df[["SubjectID"]].values]
  problem_ids = [id[0] for id in df[["ProblemID"]].values]

  EQ_scores = []
  under20_arr = []
  over120_arr = []
  for i in range(len(subject_ids)):
    one_student = eq_algo[eq_algo["SubjectID"] == subject_ids[i]]
    error_arr = one_student.loc[one_student['ProblemID'] == problem_ids[i]]
    compile_result = [i[0] for i in error_arr[['Compile.Result']].values]
    compile_message = [i[0] for i in error_arr[['CompileMessageData']].values]
    server_time_stamps = [i[0] for i in error_arr[['ServerTimestamp']].values]

    if not np.any(error_arr):
      EQ_scores.append(0.0)
      under20_arr.append(0.0)
      over120_arr.append(0.0)
    elif len(error_arr) == 1:
      EQ_scores.append(
        float(late_train[(late_train["SubjectID"] == subject_ids[i]) & (late_train["ProblemID"] == problem_ids[i])]["Label"].values[0] == False)
      )

      under20_arr.append(0.0)
      over120_arr.append(0.0)
    else:
      time_diff_arr = get_get_time_between_events(server_time_stamps)


      under20_arr.append(time_diff_arr[0])
      over120_arr.append(time_diff_arr[1])
      EQ_scores.append(get_eq_score(compile_result, compile_message, eq_1, eq_2))
  df.insert(3, "EQ", EQ_scores, True)
  df.insert(4, "%Time under 20", under20_arr, True)
  df.insert(5, "%Time over 120", over120_arr, True)
  return EQ_scores


  
def get_eq_scores(df, eq_1, eq_2):
  subject_ids = [id[0] for id in df[["SubjectID"]].values]
  problem_ids = [id[0] for id in df[["ProblemID"]].values]

  EQ_scores = []
  under20_arr = []
  over120_arr = []
  
  for i in range(len(subject_ids)):
    one_student = eq_algo[eq_algo["SubjectID"] == subject_ids[i]]
    error_arr = one_student.loc[one_student['ProblemID'] == problem_ids[i]]
    compile_result = [i[0] for i in error_arr[['Compile.Result']].values]
    compile_message = [i[0] for i in error_arr[['CompileMessageData']].values]
    server_time_stamps = [i[0] for i in error_arr[['ServerTimestamp']].values]

    if not np.any(error_arr):
      EQ_scores.append(0.0)
      under20_arr.append(0.0)
      over120_arr.append(0.0)
      
    elif len(error_arr) == 1:
      EQ_scores.append(
          float(early_train[(early_train["SubjectID"] == subject_ids[i]) & (early_train["ProblemID"] == problem_ids[i])]["Label"].values[0] == False)
      )

      under20_arr.append(0.0)
      over120_arr.append(0.0)
    else:
      time_diff_arr = get_get_time_between_events(server_time_stamps)
   

      

      under20_arr.append(time_diff_arr[0])
      over120_arr.append(time_diff_arr[1])
      EQ_scores.append(get_eq_score(compile_result, compile_message, eq_1, eq_2))
  df.insert(3, "EQ", EQ_scores, True)
  df.insert(4, "%Time under 20", under20_arr, True)
  df.insert(5, "%Time over 120", over120_arr, True)
  return EQ_scores

early_train = pd.read_csv(os.path.join(TRAIN_PATH, 'early.csv'))
get_eq_scores(early_train, 1, 3)
f_late_train = pd.read_csv(os.path.join(TRAIN_PATH, 'late.csv'))

f_X_train_base = f_late_train.copy().drop('Label', axis=1)
f_y_train = f_late_train['Label'].values

def extract_instance_features(instance, early_df):
    
    instance = instance.copy()
    subject_id = instance[PS2.SubjectID]
    early_problems = early_df[early_df[PS2.SubjectID] == subject_id]
    # Extract very naive features about the student
    # (without respect to the problem bring predicted)
    # Number of early problems attempted
    instance['ProblemsAttempted'] = early_problems.shape[0]

    # Percentage of early problems gotten correct eventually, accounting for missing assignments
    instance['PercCorrectEventually'] = np.mean( (np.concatenate( (early_problems['CorrectEventually'].values, np.full(30 - early_problems.shape[0], 0) )) ))
    
    # Median attempts made on early problems
    instance['MedAttempts'] = np.median(early_problems['Attempts'])
    # instance["streak"] = early_problems.groupby(by=["SubjectID"]).sum().loc[subject_id]["% correct b4"]
    
    # tmp = early_df[early_df["SubjectID"] == subject_id]
    # instance["correct hits"] = (tmp["prev success"] == tmp["Label"]).sum()

    instance["Mean%Time under 20"] = np.mean(early_problems["%Time under 20"])
    instance["Mean%Time over 120"] = np.mean(early_problems["%Time over 120"])
    # Max attempts made on early problems
    instance['MaxAttempts'] = np.max(early_problems['Attempts'])
    # Percentage of problems gotten correct on the first try
    instance['PercCorrectFirstTry'] = np.mean(early_problems['Attempts'] == 1)
    instance['EQ'] = np.mean(early_problems['EQ'])
    
    #instance['EQ'] =  early_problems[early_problems["EQ"] != 0].mean().fillna(0.0)["EQ"]
    # Mean of EQ scores for  each problem
    
    instance = instance.drop('SubjectID')
    return instance
def extract_features(X, early_df, scaler, is_train):
    # First extract performance features for each row
    features = X.apply(lambda instance: extract_instance_features(instance, early_df), axis=1)
    # Then one-hot encode the problem_id and append it
    problem_encoder = OneHotEncoder().fit(X[PS2.ProblemID].values.reshape(-1, 1))
    problem_ids = problem_encoder.transform(features[PS2.ProblemID].values.reshape(-1, 1)).toarray()
    # Then get rid of nominal features
    features.drop([PS2.AssignmentID, PS2.ProblemID], axis=1, inplace=True)
    # Then scale the continuous features, fitting the scaler if this is training
    feat = np.concatenate([features, problem_ids], axis=1)

    km = KMeans(n_clusters=11)
    km.fit(feat)
    clusters = km.predict(feat)
    clstrNcdr = OneHotEncoder().fit(clusters.reshape(-1, 1))
    group = clstrNcdr.transform(clusters.reshape(-1,1))
    feat = np.hstack((feat, group.toarray()))

    if is_train:
        scaler.fit(feat)
        features = scaler.transform(feat)
    
    # Return continuous and one-hot features together
    return feat

"""### Task 1
In this task, we do per-problem prediction, extracting features from performance on the 30 early problems for a given student to predict performance on each of 20 later problems. Our model should, in effect, learn the releationship between the knowledge practiced in these problems (though our naive example here won't get that far).
"""

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.model_selection import cross_validate

scaler = StandardScaler()
X_train = extract_features(f_X_train_base, early_train, scaler, False)

f_X_train = X_train

f_X_train.shape

""" # Clustering Viewer"""

errors = {}
for k in range(2,15):
  km = KMeans(n_clusters=k)
  km.fit(f_X_train)
  errors[k] = km.inertia_
plt.plot(pd.Series(errors))

from sklearn.metrics import silhouette_score
scores = {}
for k in range(2,15):
    km = KMeans(n_clusters=k)
    km.fit(f_X_train)
    clusters = km.predict(f_X_train)
    scores[k] = silhouette_score(f_X_train, clusters)
plt.plot(pd.Series(scores))

"""### Predict on the test data for the next semester (F19)

---


"""

F19_TEST_PATH = '/content/drive/Shareddrives/Learning Analytics/data/Release/F19/Test'


train_ps2 = ProgSnap2Dataset(os.path.join(F19_TEST_PATH, 'Data'))

main_table = train_ps2.get_main_table()

early_test = pd.read_csv(os.path.join(F19_TEST_PATH, 'early.csv'))
get_eq_scores(early_test, 1, 3)
late_test = pd.read_csv(os.path.join(F19_TEST_PATH, 'late.csv'))

X_test = extract_features(late_test, early_test, scaler, True)

# this is the final logistic model cell for making predictions
model = LogisticRegressionCV(max_iter=1000)
model.fit(X_train, y_train)
predictions = model.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_linear.csv')

"""# Fine Tuning Other Models"""

from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

clf = AdaBoostClassifier()
clf.fit(X_train, y_train)
predictions = clf.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_ada.csv')

clf = AdaBoostClassifier(n_estimators=13, learning_rate=0.24)
clf.fit(X_train, y_train)
train_predictions = clf.predict(X_train)
print(classification_report(y_train, train_predictions))
print('AUC: ' + str(roc_auc_score(y_train, train_predictions)))
print('Macro F1: ' + str(f1_score(y_train, train_predictions, average='macro')))

cv_results = cross_validate(clf, X_train, y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
print(f'Accuracy: {np.mean(cv_results["test_accuracy"])}')
print(f'AUC: {np.mean(cv_results["test_roc_auc"])}')
print(f'Macro F1: {np.mean(cv_results["test_f1_macro"])}')

from sklearn.tree import DecisionTreeClassifier

dlf = DecisionTreeClassifier()
val = cross_validate(dlf, X_train, y_train, cv=10, scoring=['accuracy', 'f1_macro', 'roc_auc'])
print(f'Accuracy: {np.mean(cv_results["test_accuracy"])}')
print(f'AUC: {np.mean(cv_results["test_roc_auc"])}')
print(f'Macro F1: {np.mean(cv_results["test_f1_macro"])}')

# 80, 0.5
# 10 0.46

def aclf(n_estimators, learning_rate):
  return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate)

p_grid = {
    "n_estimators" : np.arange(5,20, step=1),
    "learning_rate" : np.arange(0.01, 1.0, 0.0),
}

rs = RandomizedSearchCV(AdaBoostClassifier(), p_grid, n_iter=15, scoring="roc_auc", cv=10)
rs.fit(X_train, y_train)

results = pd.DataFrame(rs.cv_results_["params"])
results["mean score"] = rs.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

lgrg = LogisticRegressionCV(max_iter=1000)

parameter_space = {
    'Cs' : [1, 10, 100],
    'solver' : ['newton-cg', 'lbfgs'],
}
srch = RandomizedSearchCV(lgrg, parameter_space, n_jobs=-1, cv=5, n_iter=5)
srch.fit(X_train, y_train) # X is train samples and y is the corresponding labels

results = pd.DataFrame(srch.cv_results_["params"])
results["mean score"] = srch.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

lin = LogisticRegressionCV(max_iter=1000, Cs=100, solver="newton-cg")
lin.fit(X_train, y_train)
predictions = lin.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_opti_lin.csv')

from sklearn.neural_network import MLPClassifier

mlp_gs = MLPClassifier(max_iter=100)

parameter_space = {
    'hidden_layer_sizes': [(100,300,100),(200,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
}
glf = GridSearchCV(mlp_gs, parameter_space, n_jobs=-1, cv=5)
glf.fit(f_X_train, f_y_train) # X is train samples and y is the corresponding labels

mlp = MLPClassifier(activation="relu", alpha=0.05, hidden_layer_sizes=(100,300,), learning_rate="constant", solver="adam")
mlp.fit(f_X_train, f_y_train)
predictions = mlp.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_mlp.csv')

train_predictions = mlp.predict(f_X_train)
print(classification_report(f_y_train, train_predictions))
print('AUC: ' + str(roc_auc_score(f_y_train, train_predictions)))
print('Macro F1: ' + str(f1_score(f_y_train, train_predictions, average='macro')))

from sklearn.ensemble import RandomForestClassifier

f = RandomForestClassifier(max_depth=2, random_state=0)
f.fit(X_train, y_train)
predictions = f.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_forest.csv')

results = pd.DataFrame(glf.cv_results_["params"])
results["mean score"] = glf.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

from sklearn.ensemble import VotingClassifier

ada = AdaBoostClassifier() #n_estimators=13, learning_rate=0.24)
model = LogisticRegressionCV(max_iter=1000)
f = RandomForestClassifier()

softCL = VotingClassifier(
    estimators=[
                ('lr', model), ('ada', ada), ('mlp', mlp), ('rf', f)
    ], voting="soft")
softCL = softCL.fit(X_train, y_train)

predictions = softCL.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions_vote.csv')

vlf = VotingClassifier(
          estimators=[('lr',LogisticRegressionCV(max_iter=2000)), ('aboost', AdaBoostClassifier()),]
          , voting='soft')
#put the combination of parameters here 
p = [{'lr__Cs':[1,2],'aboost__n_estimators':[10,20]}]

grid = GridSearchCV(vlf,p,cv=5,scoring='roc_auc')
grid.fit(X_train,y_train)

results = pd.DataFrame(grid.cv_results_["params"])
results["mean score"] = grid.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

predictions = softCL.predict_proba(X_test)[:,1]

predictions_df = late_test.copy()
predictions_df['Label'] = predictions
predictions_df
predictions_df.to_csv('predictions.csv')

from sklearn.model_selection import GridSearchCV

def randf(n_estimators, max_depth, min_weight_fraction_leaf, max_features, bootstrap):
  return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features, bootstrap=bootstrap)

p_grid = {
    "n_estimators" : [50, 100, 150, 200],
    "max_depth" : [None, 10, 50, 100, 200],
    "min_weight_fraction_leaf" : [0.0, 0.2, 0.5, 0.75, 0.95],
    "max_features" : ["auto", "sqrt", "log2"],
    "bootstrap" : [True, False],
}

ran_for_base = randf(50, None, 0.0, "auto", True)
gs = GridSearchCV(RandomForestClassifier(), p_grid, scoring="accuracy", cv=5)
gs.fit(X_train, y_train)

results = pd.DataFrame(gs.cv_results_["params"])
results["mean score"] = gs.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

def svc_build():
  return SVC(kernel=kernel, degree=degree, coef0=coef0, probability=True)

p_grid = {
    "kernel" : ["linear", "poly", "rbf", "sigmoid"],
    "degree" : [3, 4, 6],
    "coef0" : [0.0, 0.2, 0.5, 0.75, 0.95],
}

gs = GridSearchCV(SVC(), p_grid, scoring="accuracy", cv=5)
gs.fit(X_train, y_train)

results = pd.DataFrame(gs.cv_results_["params"])
results["mean score"] = gs.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)

def aclf(n_estimators, learning_rate, base_estimator):
  return AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                            base_estimator=base_estimator)
nval = knc(n_neighbors=9, weights="uniform")

p_grid = {
    "n_estimators" : [10, 20, 50, 100, 150, 200],
    "learning_rate" : [1.0, 0.5, 2, 0.001, 0.25, 0.1, 0.3, 0.95],
}

gs = GridSearchCV(AdaBoostClassifier(), p_grid, scoring="accuracy", cv=5)
gs.fit(X_train, y_train)

results = pd.DataFrame(gs.cv_results_["params"])
results["mean score"] = gs.cv_results_["mean_test_score"]
results.sort_values(by="mean score", ascending=False)
