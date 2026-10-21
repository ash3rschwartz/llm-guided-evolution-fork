# Preprocessing from our earlier ML / MOGP assignments (Milestone 2 pipeline).
# Reads data/train.csv and writes data/processed_train.csv for LLM-GE.
#
# Steps (same as the notebook):
#   1. drop Name, Ticket, Cabin
#   2. PassengerId is an identifier, not a feature -> dropped (the notebook made it the index)
#   3. fill missing Age and Fare with the column mean, Embarked with the mode
#   4. encode Embarked {C:0, Q:1, S:2} and Sex {male:0, female:1}
#
# The StandardScaler from Milestone 2 is applied in eval.py, because it must be
# fit on the training rows only (fitting it here would leak validation data).
#
# Row order is NOT changed, so train_test_split in eval.py picks the same
# rows as the notebook did.
import pandas as pd

if __name__ == '__main__':
    train_data = pd.read_csv('data/train.csv')

    train_data = train_data.drop(columns=['Name', 'Ticket', 'Cabin', 'PassengerId'])

    nan_map = {
        'Age': train_data['Age'].mean(),
        'Fare': train_data['Fare'].mean(),
        'Embarked': train_data['Embarked'].mode()[0],
    }
    train_data = train_data.fillna(value=nan_map)

    train_data['Embarked'] = train_data['Embarked'].map({'C': 0, 'Q': 1, 'S': 2}).astype(int)
    train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1}).astype(int)

    print(train_data.head())
    print(train_data.shape)
    # index=False: do not write the row index as an extra (fake) feature column
    train_data.to_csv('data/processed_train.csv', index=False)
