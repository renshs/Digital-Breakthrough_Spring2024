import pandas as pd
import pickle
import numpy as np
import re
import fasttext

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
            6: "Технические неполадки",
            7: "Прощания",
            8: "Эмоции",
            9: "Выполнено",
            10: "Понимание",
            11: "Мусор",
            12: "Прощания",
            13: "Выполнено",
            14: "Орг. моменты",
            15: "Выполнено",
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
