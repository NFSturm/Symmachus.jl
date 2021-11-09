from pathlib import Path
from numpy import savez_compressed
from sentence_transformers import SentenceTransformer

if __name__ == '__main__':

    # Portuguese Sentence Transformer (PT_MODEL)
    pt_model = SentenceTransformer("ricardo-filho/bert-portuguese-cased-nli-assin-assin-2")

    # Multilingual Sentence Transformer (ML_MODEL)
    ml_model = SentenceTransformer("distiluse-base-multilingual-cased-v1")

    # Neural Mind Transformer (NM_MODEL)
    nm_model = SentenceTransformer("neuralmind/bert-base-portuguese-cased")

    sdg_text_files = list(Path("./data/sdg_text").iterdir())

    model_encodings = []

    for model in [pt_model, ml_model, nm_model]:
        
        encodings = []
        
        for file in sdg_text_files:
            with open(file) as f:
                text = f.readlines()
                clean_text = ' '.join([line.strip() for line in text])
                text_encoded = model.encode(clean_text)
                encodings.append(text_encoded)

        model_encodings.append(encodings)        

    #model_labels = ["pt_model", "ml_model", "nm_model"]

    for count, encodings in enumerate(model_encodings[0]):
        savez_compressed(f"./src_py/pt_arrays/sdg{count + 1}.npz", encodings)

    for count, encodings in enumerate(model_encodings[1]):
        savez_compressed(f"./src_py/ml_arrays/sdg{count + 1}.npz", encodings)

    for count, encodings in enumerate(model_encodings[2]):
        savez_compressed(f"./src_py/nm_arrays/sdg{count + 1}.npz", encodings)



