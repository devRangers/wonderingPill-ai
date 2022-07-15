from sklearn.model_selection import train_test_split
import pandas as pd

def get_train_valid_test(data_df, valid_size=0.1, random_state=216):
  data_path = data_df['path'].values
  data_label = pd.get_dummies(data_df['label']).values

  X_train, X_test, y_train, y_test = train_test_split(data_path, data_label, test_size=valid_size, stratify=data_label, random_state=216)
  X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=(valid_size / (1-valid_size)), stratify=y_train, random_state=216)

  return X_train, X_valid, X_test, y_train, y_valid, y_test
