from django.shortcuts import render
from transformers import MarianMTModel, MarianTokenizer # type: ignore
from pydantic import BaseModel

# Create your views here.
class TextInput(BaseModel):
    content: str

model_name = 'Helsinki-NLP/opus-mt-ar-en'
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

def formInfo(request):
    if request.method == 'POST':
        Text = request.POST['inputText']
        trans = translate_text(Text)
        return render(request,'ui.html',{'ans' : trans})
    return render(request, 'ui.html')

def translate_text(text: str):   
    # Tokenize the input text
    tokenized_input = tokenizer(text, return_tensors="pt", padding=True)

    # Generate the translation
    translated_tokens = model.generate(**tokenized_input)

    # Decode the translated text
    translated_text = tokenizer.decode(translated_tokens[0], skip_special_tokens=True)
    print(translated_text)
    return translated_text
