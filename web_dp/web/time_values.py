import numpy as np
import pandas as pd
from collections import Counter
import re
from matplotlib import pyplot as plt
import seaborn as sns
import statistics


def insults(text):
    insensitive_hippo = re.compile(re.escape("спидр"), re.IGNORECASE)
    x = insensitive_hippo.sub("", text)
    template = "(?iu)(?<![а-яё])(?:(?:(?:у|[нз]а|(?:хитро|не)?вз?[ыьъ]|с[ьъ]|(?:и|ра)[зс]ъ?|(?:о[тб]|п[оа]д)[ьъ]?|(?:\S(?=[а-яё]))+?[оаеи-])-?)?(?:[её](?:б(?!о[рй]|рач)|п[уа](?:ц|тс))|и[пб][ае][тцд][ьъ]).*?|(?:(?:н[иеа]|ра[зс]|[зд]?[ао](?:т|дн[оа])?|с(?:м[еи])?|а[пб]ч)-?)?ху(?:[яйиеёю]|л+и(?!ган)).*?|бл(?:[эя]|еа?)(?:[дт][ьъ]?)?|\S*?(?:п(?:[иеё]зд|ид[аое]?р|ед(?:р(?!о)|[аое]р|ик)|охую)|бля(?:[дбц]|тс)|[ое]ху[яйиеё]|хуйн).*?|(?:о[тб]?|про|на|вы)?м(?:анд(?:[ауеыи](?:л(?:и[сзщ])?[ауеиы])?|ой|[ао]в.*?|юк(?:ов|[ауи])?|е[нт]ь|ища)|уд(?:[яаиое].+?|е?н(?:[ьюия]|ей))|[ао]л[ао]ф[ьъ](?:[яиюе]|[еёо]й))|елд[ауые].*?|ля[тд]ь|(?:[нз]а|по)х)(?![а-яё])"
    x = x if len(x) < 100 else x[:100]
    return 1 if re.search(template, x) else 0



def raspredelenie(csv_name):
    df = pd.read_csv(csv_name)

    if df.shape[0] >= 1000:
        df = df.head(1000)

    df.dropna(how="all", inplace=True)
    df["Текст сообщения"] = df["Текст сообщения"].fillna("")
    df["clean_text"] = df["Текст сообщения"].apply(
        lambda x: re.sub(r"[0-9-TZ:.;\n=\[\],]", "", str(x))
    )
    df["url"] = df["Текст сообщения"].apply(
        lambda x: (
            1
            if re.search(
                "http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
                x,
            )
            else 0
        )
    )
    df["bad_word"] = df["Текст сообщения"].apply(insults)


    # Группировка по ID, потом по дате начала урока
    grouped_df = df.groupby(["ID урока", "Дата старта урока"], as_index=False)
    new_df1 = grouped_df.agg(list)

    new_df1.url = new_df1.url.apply(lambda x: sum(x))
    new_df1.bad_word = new_df1.bad_word.apply(lambda x: sum(x))

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
        urls = new_df1.url[i]
        bad_words = new_df1.bad_word[i]

        counter = {k: 0 for k in range(messages.max())}

        [counter.update({k: v}) for k, v in Counter(messages).items()]
        y = list(counter.values())
        result_y.append((y, f"{ids}_{strart_times}", time_last, text, urls, bad_words))

    return result_y


def plot_mpr(values):
    mpr_ref = np.load("web_dp/web/mpr.npy")
    mpr = sum(values[0]) / values[2]
    sns.displot(mpr_ref, kde=True)
    plt.axvline(mpr, color="red", linewidth=2, label='Активность на уроке')

    plt.axvline(
        np.percentile(mpr_ref, 30), color="green", alpha=0.5, linewidth=2, label='Границы CI'
    )
    plt.axvline(
        np.percentile(mpr_ref, 70), color="green", alpha=0.5, linewidth=2
    )
    plt.legend()
    plt.xlim(-0.05, 4)
    plt.title('Messages per minute')
    plt.savefig(f"web_dp/static/charts/{values[1]}_mpr.png", bbox_inches='tight')
    plt.close()


def plot_activities(values):
    
    act_ref = np.load("web_dp/web/acrivities_counts.npy")
    activity_rate = len(
        [i for i in values[0] if i > statistics.mode(values[0])]
    )
    sns.displot(act_ref, kde=True)
    plt.axvline(activity_rate, color="red", linewidth=2, label='Активность на уроке')

    plt.axvline(
        np.percentile(act_ref, 30), color="green", alpha=0.5, linewidth=2, label='Границы CI'
    )
    plt.axvline(
        np.percentile(act_ref, 70), color="green", alpha=0.5, linewidth=2
    )
    plt.legend()
    plt.title('Activities number')
    plt.savefig(f"web_dp/static/charts/{values[1]}_act.png", bbox_inches='tight')
    plt.close()


#     plt.xlim(-.05, 4)
if __name__ == "__main__":
    values = raspredelenie("train_GB_KachestvoPrepodovaniya.csv")
    print(values[21])
