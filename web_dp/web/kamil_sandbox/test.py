import pandas as pd
import pickle
import numpy as np
import re
import fasttext
from collections import Counter
from matplotlib import pyplot as plt
import seaborn as sns
import statistics

em_reload = pickle.load(open("web_dp/web/em_25_on_50pca_from_fasttext.pkl",'rb'))
pca_reload = pickle.load(open("web_dp/web/pca_fasttext_300_to_50.pkl",'rb'))
model_path = 'web_dp/web/ft_native_300_ru_wiki_lenta_remstopwords.bin'
model = fasttext.load_model(model_path)

clusters = {0: "code/translit",
            1: "Восхищения",
            2: "Числа",
            3: "Благодарности",
            4: "Приветствия",
            5: "Вопросы",
            6: "У кого-то что-то",
            7: "Прощания",
            8: "Эмоции",
            9: "Что-то сделано",
            10: "Понимание",
            11: "Мусор",
            12: "Прощания",
            13: "я что-то сделал",
            14: "Орг. моменты",
            15: "я что-то сделал",
            16: "Орг. моменты",
            17: "Восхищения",
            18: "Пожелания",
            19: "Эмоции",
            20: "Эмоции",
            21: "Мусор",
            22: "Ответы",
            23: "Мусор",
            24: "Мусор"}

def clasterize(text_list):
    df_list = {}
    for i in range(len(text_list)):
        df = pd.DataFrame(text_list[i], columns=["Текст сообщения"])
        # print(df)
        text_vectorized = df["Текст сообщения"].apply(lambda x: re.sub(r'[0-9-TZ:.;\n=\[\],]', '', str(x))).apply(lambda x: model.get_sentence_vector(x.lower())).values
        text_vectorized = np.array([text.tolist() for text in text_vectorized])
        result_new = pca_reload.transform(text_vectorized)
        predicted = em_reload.predict(result_new)
        df["cluster"] = predicted
        # print(predicted)
        df["cluster"] = df["cluster"].replace(clusters)
        df_list[i] = {"df": df,
                    "code_count": (df["cluster"] == "code/translit").sum(),
                    "Восхищения_count": (df["cluster"] == "Восхищения").sum(),
                    "Числа_count": (df["cluster"] == "Числа").sum(),
                    "Благодарности_count": (df["cluster"] == "Благодарности").sum(),
                    "Приветствия_count": (df["cluster"] == "Приветствия").sum(),
                    "Вопросы_count": (df["cluster"] == "Вопросы").sum(),
                    "У кого-то что-то_count": (df["cluster"] == "У кого-то что-то").sum(),
                    "Прощания_count": (df["cluster"] == "Прощания").sum(),
                    "Эмоции_count": (df["cluster"] == "Эмоции").sum(),
                    "Done_count": (df["cluster"] == "Что-то сделано").sum(),
                    "Понимание_count": (df["cluster"] == "Понимание").sum(),
                    "Мусор_count": (df["cluster"] == "Мусор").sum(),
                    "я что-то сделал_count": (df["cluster"] == "я что-то сделал").sum(),
                    "Мусор_count": (df["cluster"] == "Мусор").sum(),
                    "кого-то не будет_count": (df["cluster"] == "кого-то не будет").sum(),
                    "Восхищения_count": (df["cluster"] == "Восхищения").sum(),
                    "Пожелания_count": (df["cluster"] == "Пожелания").sum(),
                    "Ответы_count": (df["cluster"] == "Ответы").sum()}
    return df_list






def raspredelenie(csv_name):
    df = pd.read_csv(csv_name)
    # Группировка по ID, потом по дате начала урока
    grouped_df = df.groupby(["ID урока", "Дата старта урока"], as_index=False)
    new_df1 = grouped_df.agg(list)

    def check_datetime_format(text):
        """
        Проверяет, соответствует ли текст формату 'YYYY-MM-DD HH:MM:SS'.
        """
        pattern = r"^\d{4}-\d{2}-\d{2}\s\d{2}:\d{2}:\d{2}$"
        return bool(re.match(pattern, text))

    result_column = []
    time_last_mess = []
    for i in range(len(new_df1)):
        date_message = list(
            new_df1[new_df1["ID урока"] == new_df1["ID урока"][i]][
                "Дата сообщения"
            ]
        )[0]

        date_message = [value for value in date_message if not pd.isna(value)]

        times = []
        # делю время откправки сообщения от даты
        count = 0
        for datetime_str in date_message:
            if check_datetime_format(datetime_str):
                date, time = datetime_str.split()  # Разделение по пробелу
                times.append(time)
            else:
                del date_message[count]
            count += 1

        times_in_seconds = []
        # перевожу формат часы минуты секунды в секунды(время отправки)
        for time_str in times:
            hours, minutes, seconds = map(
                int, time_str.split(":")
            )  # Разделение и преобразование в int
            total_seconds = hours * 3600 + minutes * 60 + seconds
            times_in_seconds.append(total_seconds)

        # print(times_in_seconds)

        times_start_web = list(
            new_df1[new_df1["ID урока"] == new_df1["ID урока"][i]][
                "Дата старта урока"
            ]
        )[0].split(", ")
        hours_s, minutes_s = map(int, times_start_web[1].split(":"))
        seconds_s = hours_s * 3600 + minutes_s * 60

        raznica = []
        for j in times_in_seconds:
            raznica.append(j - seconds_s)
        result_column.append(raznica)

        time_last_mess.append(raznica[-1] / 60)

    new_df1["Разница между сообщениями и началом вебинара"] = result_column

    def divide_by_120(list_of_numbers):
        return [int(num / 300) for num in list_of_numbers]

    new_df1["Разница между сообщениями и началом вебинара"] = new_df1[
        "Разница между сообщениями и началом вебинара"
    ].apply(divide_by_120)

    result_y = []

    for i in range(len(new_df1)):
        messages = np.array(
            new_df1["Разница между сообщениями и началом вебинара"][i]
        )
        ids = str(new_df1["ID урока"][i])
        strart_times = str(new_df1["Дата старта урока"][i])
        time_last = time_last_mess[i]
        text = new_df1["Текст сообщения"][i]

        counter = {k: 0 for k in range(messages.max())}

        [counter.update({k: v}) for k, v in Counter(messages).items()]
        y = list(counter.values())
        result_y.append((y, f"{ids}_{strart_times}", time_last, text))

    return result_y


def plot_mpr(values):
    mpr_ref = np.load("web_dp/web/mpr.npy")
    mpr = sum(values[0]) / values[2]
    sns.displot(mpr_ref, kde=True)
    plt.axvline(mpr, color="red", linewidth=2)

    plt.axvline(
        np.percentile(mpr_ref, 30), color="green", alpha=0.5, linewidth=2
    )
    plt.axvline(
        np.percentile(mpr_ref, 70), color="green", alpha=0.5, linewidth=2
    )

    plt.xlim(-0.05, 4)
    plt.title('Messages per minute')
    plt.savefig(f"web_dp/static/charts/{values[1]}_mpr.png", bbox_inches='tight')


def plot_activities(values):
    
    act_ref = np.load("web_dp/web/acrivities_counts.npy")
    activity_rate = len(
        [i for i in values[0] if i > statistics.mode(values[0])]
    )
    sns.displot(act_ref, kde=True)
    plt.axvline(activity_rate, color="red", linewidth=2)

    plt.axvline(
        np.percentile(act_ref, 30), color="green", alpha=0.5, linewidth=2
    )
    plt.axvline(
        np.percentile(act_ref, 70), color="green", alpha=0.5, linewidth=2
    )
    plt.title('Activities number')
    plt.savefig(f"web_dp/static/charts/{values[1]}_act.png", bbox_inches='tight')


if __name__ == "__main__":
    values = raspredelenie("/../../../jupyter/train_GB_KachestvoPrepodovaniya.csv")
    print(values[21])
