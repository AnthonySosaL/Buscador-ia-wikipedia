from transformers import DistilBertTokenizer, DistilBertForQuestionAnswering
import torch
import wikipedia

wikipedia.set_lang("es")

def search_wikipedia(query):
    search_results = wikipedia.search(query)

    if not search_results:
        print("No se encontraron resultados en Wikipedia.")
        return None

    page = wikipedia.page(search_results[0])
    return page.content

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')

if torch.cuda.is_available():
    device = torch.device("cuda")
    print("Usando la GPU:", torch.cuda.get_device_name(0))
else:
    device = torch.device("cpu")
    print("Usando la CPU")

model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad').to(device)

question = input("Por favor, escribe tu pregunta: ")

text = search_wikipedia(question)

if text:
    paragraphs = [para for para in text.split('\n') if para.strip()]
    full_context = " ".join(paragraphs)

    inputs = tokenizer(question, full_context, return_tensors='pt', max_length=512, truncation=True, padding='max_length')

    inputs = {key: tensor.to(device) for key, tensor in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    start_scores = outputs.start_logits
    end_scores = outputs.end_logits
    answer_start = torch.argmax(start_scores)
    answer_end = torch.argmax(end_scores)
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end+1].cpu(), skip_special_tokens=True)

    if answer:
        print("Respuesta:", answer)
    else:
        print("No se encontro una respuesta clara. Aqui hay informacion adicional de Wikipedia:")
        print(text)
