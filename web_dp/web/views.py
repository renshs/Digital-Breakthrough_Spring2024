import uuid
import web.clastetize as clust

from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from web_dp.form import InputFileForm
from web_dp import settings
import os
import web.time_values as time_values
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
# Create your views here.


# id -- id урока
# date -- дата проведения
# result -- здесь все выводы
# msg - здесь key : list -- key -название кластера, list -- все сообщения
# charts -- графики

# TODO: модельку сюдааааааа
def model_handler(file_name, flag = True):
    data = time_values.raspredelenie(f"web_dp/uploads/{file_name}")
    id_list = []
    texts = []
    links_count = []
    absc_count = []
    for dat in data:
        # if flag:
        #     time_values.plot_mpr(dat)
        #     time_values.plot_activities(dat)
        id_list.append(("charts/" + dat[1] + '_act.png', "charts/" + dat[1] + '_mpr.png'))
        texts.append(dat[3])
        links_count.append(dat[4])
        absc_count.append(dat[5])
    texts = clust.clasterize(texts)
    # print("\n"*3)
    # print(texts[0]["df"])
    return id_list, data, texts, flag, links_count, absc_count


# TODO: фильтр (не кофе)
def filter_res(items, checkboxes):
    pass


def index(request):
    if request.method == 'POST':
        print(request.POST)
        charts_flag = request.POST.get('chart_bool') 

        form = InputFileForm(request.POST, request.FILES)
        
        if form.is_valid():
            file = request.FILES["file"]
            fs = FileSystemStorage()
            file_name = fs.save(f"{uuid.uuid4()}_{file.name}", file)

            id_list, res, table_items, flag, links_count, absc_count = model_handler(file_name, charts_flag)

            # filter_res(res, checkboxes)
            # print(links_count, absc_count)
            table_items_2 = []
            for i in range(len(table_items)):
                # print(table_items[i])
                table_items_2.append(list(table_items[i]["df"].apply(lambda x: (x["Текст сообщения"], x["cluster"]), axis=1)))
                # print(table_items[i])
            ids = [i[0][7:-8] for i in id_list]
            for_score = []
            for i in range(len(table_items)):
                sum_clus = len(table_items[i]["df"])
                for_score.append([table_items[i]["Ответы_count"]/sum_clus,
                                  table_items[i]["Понимание_count"]/sum_clus,
                                  table_items[i]["Благодарности_count"]/sum_clus,
                                  table_items[i]["code_count"]/sum_clus,
                                  table_items[i]["Восхищения_count"]/sum_clus,
                                  links_count[i]/sum_clus, absc_count[i]/sum_clus])
                # print(table_items[i])
            for_score = np.array(for_score)
            score = ((for_score.T[0] + for_score.T[1] + for_score.T[2] + for_score.T[4]) * 80 + for_score.T[3] * 50 + for_score.T[5] * 15 - for_score.T[6] * 100).T * 3.3
            score = score.astype(np.uint32)
            score[score > 100] = 100
            # figure = plt.plot([1, 2 ,3], [3, 1, 3])
            # plt.close()
            # plt.savefig("none.png")
            sns.lineplot(pd.DataFrame({"Номер урока": range(1, len(table_items) + 1), "Оценка": score}), x ="Номер урока", y="Оценка")
            plt.title("Изменение Оценки")
            plt.savefig("web_dp/static/score1.png", bbox_inches='tight')
            plt.close()
            sns.displot(score)
            plt.title('Распределение оценок')
            plt.savefig("web_dp/static/dis.png", bbox_inches='tight')
            plt.close()
            for dat in res:
                if flag:
                    time_values.plot_mpr(dat)
                    time_values.plot_activities(dat)
            return render(request, 'results.html', 
                          context={'id_list': id_list, 
                                   "table_items": table_items_2, 
                                   "ranga": list(range(len(table_items))), 
                                   "zipped": zip(id_list, table_items_2, score),
                                   "categories": list(set(clust.clusters.values())),
                                   "flag": flag,
                                   "ids": ids

                                #    "score": for_score
                                   })

    form = InputFileForm()
    # print(list(set(clust.clusters.values())))
    return render(request, 'index.html', context={'form': form, 'categories': list(set(clust.clusters.values()))})


def about(request):
    return render(request, 'about.html')



file_names = [f for f in os.listdir(settings.MEDIA_ROOT) if os.path.isfile(os.path.join(settings.MEDIA_ROOT, f))]

print(file_names)
