from cr_learn import beibei as b

data = b.load()

trn_buy_df = data['trn_buy']
print(trn_buy_df.head())

trn_buy_dataset = data['datasets']['trn_buy']
print(f"Number of users: {trn_buy_dataset.get_user_count()}")

all_dfs = data['all_dataframes']