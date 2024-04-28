from django import forms
from web import models


class InputFileForm(forms.Form):
    file = forms.FileField(required=True, label='Выберите файл')

    class Meta:
        model = models.Document
        fields = ['file']
        

