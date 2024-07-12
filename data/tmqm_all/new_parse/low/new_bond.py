import os
import pandas as pd

list_files = os.listdir("./")
for i in list_files: 
    print(i)
    df = pd.read_pickle(i)
    # check that there is a column "bonds", else pass
    if "bonds" in df.columns:
        bond_list = df.bonds.tolist()
        #bond_list_new = []
        rows_to_remove = []
        for ind, bond_list_temp in enumerate(bond_list):
                if len(bond_list_temp) == 0:
                    # remove row if there are no bonds
                    rows_to_remove.append(ind)
                else:
                    bond_list_temp.append(bond_list_temp[0])
        # remove rows with no bonds
        print("removing rows: ", rows_to_remove)
        df = df.drop(df.index[rows_to_remove])
        #df["bonds"] = bond_list_new
        df.to_pickle(i)
    else:
        pass
