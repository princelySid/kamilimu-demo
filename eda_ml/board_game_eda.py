# %%
import ast
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import MinMaxScaler

# %%
path = "/Users/princelysid/projects/personal/kamilimu/board_games"
raw_ratings = pd.read_csv(f"{path}/ratings.csv")
raw_details = pd.read_csv(f"{path}/details.csv")
# %%
raw_ratings.head()
# %%
# We started by looking at the ratings
# We filtered num, url, thumbnail  columns because they were found to not add any information to
# our data. We also narrowed our data to the years 1963 and 2022, this was an abbitratary decision

fil_ratings = raw_ratings[
    ["id", "name", "year", "rank", "average", "bayes_average", "users_rated"]
]
fil_ratings = fil_ratings[(fil_ratings["year"] >= 1963) & (fil_ratings["year"] <= 2022)]

# %%
fil_ratings[fil_ratings["name"] == "Decipher"]
# %%
# Number of ratings has been increasing over the year but the median number of reviews has
# remanined relative stable, which means that more games has not reduced the number of
# reviews per game
year_counts = (
    fil_ratings["year"].value_counts().reset_index(name="count").sort_values(by="index")
)
fig, ax = plt.subplots()
ax.bar(x=year_counts["index"], height=year_counts["count"])

fil_ratings.groupby("year")["users_rated"].median().reset_index().plot.line(
    x="year", y="users_rated"
)
fil_ratings.groupby("year")["users_rated"].mean().reset_index().plot.line(
    x="year", y="users_rated"
)
# %%
# We also want to look at the two averages
# https://arpitbhayani.me/blogs/bayesian-average


fil_ratings["bayes_average"].plot.hist()
fil_ratings["average"].plot.hist()
# %%
# %%
# Wanted to see  how people name their games in terms of how many words
fil_ratings["name"].str.split().str.len().plot.hist()

# %%
# Finally we need to look at our details data
# Most games are made for a mininum of 2 players, and a max 6
# Most games have a min play time of under an hour and max of one and a half
# majority of the game are for 10 and over.
raw_details.head()
raw_details.shape
raw_details.describe()
# %%
# We apply some of the same filters here we did to our other data
fil_details = raw_details[
    [
        "id",
        "primary",
        "description",
        "yearpublished",
        "minplayers",
        "maxplayers",
        "playingtime",
        "minplaytime",
        "maxplaytime",
        "minage",
        "boardgamecategory",
        "boardgamemechanic",
        "boardgamefamily",
        "boardgameexpansion",
        "boardgameimplementation",
        "boardgamedesigner",
        "boardgameartist",
        "boardgamepublisher",
        "owned",
        "trading",
        "wanting",
        "wishing",
    ]
]
fil_details = fil_details[
    (fil_details["yearpublished"] >= 1963) & (fil_details["yearpublished"] <= 2022)
]
# %%
fil_details["minplayers"].value_counts().reset_index(name="count")
fil_details["minplayers"].plot.hist()
# %%
# There's a descision to be made here on whether we should filter on this too but I'm leaving it
fil_details["maxplayers"].plot.hist()
# %%
a = fil_details["maxplayers"].value_counts().reset_index(name="count")
a = a[a["index"] < 100]
a.sort_values("index", inplace=True)
fig, ax = plt.subplots()
ax.plot(a["index"], a["count"])
# %%
fil_details["playingtime"].plot.hist()
# %%
a = (
    fil_details["playingtime"]
    .value_counts()
    .reset_index(name="count")
    .sort_values("index")
)
a = a[a["index"] < 150]
fig, ax = plt.subplots()
ax.plot(a["index"], a["count"])


# %%
def plot_value_counts(df: pd.DataFrame, column: str, number_of_values: int):
    a = df[column].value_counts().reset_index(name="count").sort_values("index")
    total = a.shape[0]
    a = a.iloc[:number_of_values]
    fig, ax = plt.subplots()
    ax.plot(a["index"], a["count"])
    plt.title(f"{column} with {number_of_values} of {total} values ")


# %%
plot_value_counts(fil_details, "minplaytime", 50)
plot_value_counts(fil_details, "maxplaytime", 50)
# %%
plot_value_counts(fil_details, "minage", 50)

# %%
text_data = fil_details[
    [
        "primary",
        "description",
        "boardgamecategory",
        "boardgamemechanic",
        "boardgamefamily",
        "boardgameexpansion",
        "boardgameimplementation",
        "boardgamedesigner",
        "boardgameartist",
        "boardgamepublisher",
    ]
]

# %%
# Find the nas
text_data.isna().sum()


# %%
# count number of columns we import ast for this
def load_list(row):
    try:
        return ast.literal_eval(row)
    except ValueError:
        return


def get_unique_values_list(df, column):
    val = set()
    for row in df[column]:
        data = load_list(row)
        if data:
            data = set(data)
            val = val | data
    return val


def count_in_list(df, column, unique_values):
    counts = {value: 0 for value in unique_values}
    for row in df[column]:
        data = load_list(row)
        if data:
            for l in data:
                counts[l] += 1
    return counts


def graph_top_x(df: pd.DataFrame, count_col: str, cat_col: str, x: int):
    a = df.sort_values(by=count_col, ascending=False)
    a = a.iloc[:x]
    a = a.sort_values(by=count_col)
    fig, ax = plt.subplots()
    ax.barh(a[cat_col], width=a[count_col])
    plt.title(f"{cat_col}:top {x}")


# %%
unique_values = get_unique_values_list(text_data, "boardgamecategory")
count_values = count_in_list(text_data, "boardgamecategory", unique_values)
a = pd.DataFrame(count_values.items(), columns=["boardgamecategory", "count"])
graph_top_x(a, "count", "boardgamecategory", 20)


# %%
def graph_cat_top_x(df: pd.DataFrame, cat_col, x):
    unique_values = get_unique_values_list(df, cat_col)
    count_values = count_in_list(df, cat_col, unique_values)
    a = pd.DataFrame(count_values.items(), columns=[cat_col, "count"])
    graph_top_x(a, "count", cat_col, x)


# %%
graph_cat_top_x(text_data, "boardgamecategory", 20)
graph_cat_top_x(text_data, "boardgamemechanic", 20)
graph_cat_top_x(text_data, "boardgamefamily", 20)
graph_cat_top_x(text_data, "boardgameexpansion", 20)
graph_cat_top_x(text_data, "boardgameimplementation", 20)
graph_cat_top_x(text_data, "boardgamedesigner", 20)
graph_cat_top_x(text_data, "boardgameartist", 20)
graph_cat_top_x(text_data, "boardgamepublisher", 20)

# %%
x_all = fil_ratings[["average", "bayes_average", "users_rated"]]
y_all = fil_ratings["rank"]

minmax = MinMaxScaler()
x_all_scaled = minmax.fit_transform(x_all)
# %%
minmax.transform(
    pd.DataFrame.from_dict(
        {"average": 8, "bayes_average": 7, "users_rated": 1000}, orient="index"
    ).T
)
# %%
reg = linear_model.LinearRegression()
reg.fit(x_all_scaled, y_all)
# %%
pickle.dump(reg, open("our_model.kamilimu", "wb"))
pickle.dump(minmax, open("scaler.kamilimu", "wb"))
# %%
data = pd.DataFrame.from_dict(
    {"average": 7, "bayes_average": 5, "users_rated": 7000}, orient="index"
).T
scaled = minmax.transform(data)
reg.predict(scaled)
# %%
