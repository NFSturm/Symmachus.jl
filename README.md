# Symmachus.jl

> We see the same stars, the sky is shared by all, the same world surrounds us. What does it matter what wisdom a person uses to seek for the truth? - Quintus Aurelius Symmachus (342 - 402/3 CE)

![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

💡 This project is realized as a Master thesis at Nova School of Business and Economics, Lisbon under the supervision of Prof. Leid Zejnilovic. 

📄 Symmachus is a semi-supervised Machine Learning system. The idea is to use a small, but expressive dataset containing phrases containing political convictions and normative statements from the Portuguese parliament and gradually building a bigger dataset, which can then be used for analysis. 

💾 The data is derived from [Debates Parlamentares](https://debates.parlamento.pt), for the discourse aspect and the [Assembleia da República](https://www.parlamento.pt/Cidadania/Paginas/DAatividadeDeputado.aspx) for deputies' activities. The ETL process was implemented in Python and is stored in another repository at the Data Science Knowledge Center.

⌨️ There are several components to this repository:

1. There is *SymmachusCore*, which is a way of embedding a sentence by considering both the immediate surroundings of the sentence root on a token level and the neighbouring sentences of the one in question. To this end it uses FastText embeddings provided by `Embeddings.jl`. 
2. A self-training labelling machine based on `XGBoost.jl` is used to generate a dataset large enough to reliably train a broadcasting device on it. Using this device, the entire dataset is labelled in order to make it ready for analysis.
3. To then conduct the analysis, a re-implementation of *Rake (Rapid Automatic Keyword Extraction)* (described by e.g. *Berry & Kogan* (2010)) is employed. This serves to compare the two datasets on a sentence-by-sentence level. 



