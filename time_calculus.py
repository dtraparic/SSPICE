from __future__ import annotations
import datetime
from pathlib import Path
from typing import Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demo1():
    v = "Dune_0"
    input_dir = Path("E:/ICE_CUBED_RESULTS/frames/" + v)
    suffix = "_BIREFmxt3"
    output_dir = Path(f"E:/DATA fast label retrain biref/corrected_masks/{v}{suffix}")
    all_paths_of_time_logs_file = list(Path(output_dir).glob("*.txt"))
    df_list = []
    for path in all_paths_of_time_logs_file:
        tmp_df = pd.read_csv(path, header=None, index_col=None)
        print(tmp_df.shape)
        df_list.append(tmp_df)
    new_df = pd.concat(df_list, ignore_index=True)
    print(new_df)
    print(new_df.shape)
    new_df.to_csv(f"E:/DATA fast label retrain biref/corrected_masks/truc.csv", index=True)

def concat_all_txt_files(folder: Path):
    import os
    fichiers = sorted(list(folder.glob("time_log_*.txt")))
    contenu_total = ""
    for fichier in fichiers:
        chemin_complet = os.path.join(folder, fichier)
        with open(chemin_complet, 'r', encoding='utf-8') as f:
            contenu_total += f.read() + "\n"
    with open(os.path.join(folder, "fichier_concatené.txt"), 'w', encoding='utf-8') as sortie:
        sortie.write(contenu_total)
    print("Fichiers concaténés avec succès dans fichier_concatené.txt !")

def convert_one_txt_to_v1df(txtfile: Path):
    unix = int(txtfile.name.split("time_log_")[1].split(".")[0])
    df = pd.read_csv(txtfile, header=None, names=["i_frame", "timespent"])
    df["end_unix"] = unix
    df["end_day"] = datetime.datetime.fromtimestamp(unix).strftime("%Y-%m-%d")
    df["end_clock"] = datetime.datetime.fromtimestamp(unix).strftime("%H:%M:%S")
    df["i_frame"] = df["i_frame"].astype(int)
    df["i_frame"] = df["i_frame"] - 1
    return df

def delete_empty_line(txtfile: Path):
    with txtfile.open('r', encoding='utf-8') as file:
        lines = file.readlines()
    non_empty_lines = [line for line in lines if line.strip()]
    with txtfile.open('w', encoding='utf-8') as file:
        file.writelines(non_empty_lines)
    print(f"Lignes vides supprimées dans {txtfile.name} !")

def transform_txt_to_csv(txtfile: Path, csvfile: Path):
    df = pd.read_csv(txtfile, header=None, names=["i_frame", "timespent"])
    df.to_csv(csvfile, index=False)
    print(f"Fichier CSV créé : {csvfile.name}")

def remove_1_on_i_frame_column(df: pd.DataFrame):
    df["i_frame"] = df["i_frame"].apply(lambda x: x - 1)
    return df

def group_by_iframe(df: pd.DataFrame):
    """
    >>> dict_for_df = {"i_frame": [244,245,246,246,247,248,247,248,249,250,251], "timespent": [62,174,85,25,124,68,86,70,104,67,70]}
    >>> df = pd.DataFrame(dict_for_df)
    >>> group_by_iframe(df=df)
    """
    df = df.loc[df.groupby("i_frame").apply(lambda x: x.index.max())]
    # df = df.groupby("i_frame", as_index=False).max()
    return df

def check_if_yyyy_mm_dd(date: str):
    try:
        datetime.date.fromisoformat(date)
        return True
    except ValueError:
        return False

def show_stats(df: pd.DataFrame, of_day: Literal["all","today"] | str = "today"):
    assert isinstance(of_day, str)
    assert check_if_yyyy_mm_dd(of_day) or (of_day in ["all", "today"])

    if of_day == "today":
        of_day = datetime.date.today().strftime("%Y-%m-%d")

    if of_day == "all":
        df_filtered = df
    else:
        df_filtered = df[df["end_day"] == of_day]

    bins = np.arange(0, df_filtered["timespent"].max()+10, 10)
    plt.hist(df_filtered["timespent"], bins=bins, cumulative=False, histtype='bar')
    plt.xlabel('Time Spent (sec, step of 10)')
    plt.ylabel('Occurences')
    plt.yscale('log', base=2)
    plt.yticks([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024], ['1', '2', '4', '8', '16', '32', '64', '128', '256', '512', '1024'])
    mean = df_filtered["timespent"].mean()
    std = df_filtered["timespent"].std()
    median = df_filtered["timespent"].median()
    plt.axvline(mean, color='tab:red', linestyle='dashed', linewidth=2)
    plt.axvline(median, color='tab:orange', linestyle='dashed', linewidth=2)
    # plt.axvline(mean + std, color='tab:cyan', linestyle='dashed', linewidth=2)
    # plt.axvline(mean - std, color='tab:cyan', linestyle='dashed', linewidth=2)

    plt.title(f'Time Spent Distribution (on {df_filtered.shape[0]} frames on {of_day} days)\n'
              f'mean={mean:.2f}, median={median:.2f}')
    # now add legend with value of mean std median
    plt.legend([f'mean ({mean:.2f})', f'median ({median:.2f})'])

    plt.show()

def show_frames_done_per_day(df: pd.DataFrame):
    """
    Doing a wide bins plot
    """
    fig, ax = plt.subplots()
    samples = df["end_day"].value_counts(sort=False)
    samples_ordered_by_index = samples.sort_index()
    ax = samples_ordered_by_index.plot(kind="bar")
    ax.set_xlabel('day')
    ax.set_ylabel('Number of frames done')
    ax.set_title(f'Number of frames done per day from {df["end_day"].min()} to {df["end_day"].max()}')
    plt.xticks(rotation=45)
    plt.show()


if __name__ == '__main__':
    folder = Path("E:/DATA fast label retrain biref/corrected_masks/Dune_0_BIREFmxt3")
    # df = pd.read_csv(folder / f"fichier_concatené_unique_v1.csv", index_col=None)

    # concat_all_txt_files(folder=Path("E:/DATA fast label retrain biref/corrected_masks/Dune_0_BIREFmxt3"))
    # Convert v0 into format of v1, thus, adding "end_unix", "end_day" and "end_clock"
    # path_v0_csv = Path("E:/DATA fast label retrain biref/corrected_masks/Dune_0_BIREFmxt3/fichier_concatené_v0.csv")

    # path_v1_csv = Path("E:/DATA fast label retrain biref/corrected_masks/Dune_0_BIREFmxt3/fichier_concatené_v1.csv")
    # df = group_by_iframe(df=pd.read_csv(path_v1_csv, index_col=None))
    # path = Path("E:/DATA fast label retrain biref/corrected_masks/Dune_0_BIREFmxt3/fichier_concatené_unique_v1.csv")
    # df.to_csv(path, index=False)
    # show_stats(df, of_day="today")

    df_v0 = pd.read_csv(folder / f"fichier_concatené_v0.csv", index_col=None)
    df_v1 = pd.read_csv(folder / f"fichier_concatené_v1.csv", index_col=None)
    df = pd.concat([df_v0, df_v1], ignore_index=True)
    group_by_iframe(df=df)
    show_stats(df, of_day="all")
    show_stats(df, of_day="today")
    show_frames_done_per_day(df)


