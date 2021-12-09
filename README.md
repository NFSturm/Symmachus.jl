# Symmachus.jl

> We see the same stars, the sky is shared by all, the same world surrounds us. What does it matter what wisdom a person uses to seek for the truth? - Quintus Aurelius Symmachus (342 - 402/3 CE)

![Julia](https://img.shields.io/badge/-Julia-9558B2?style=for-the-badge&logo=julia&logoColor=white)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)

💡 This project is realized as a Master thesis at Nova School of Business and Economics, Lisbon under the supervision of Prof. Leid Zejnilovic. 

📄 Symmachus implements the functionalities a semantic search engine for Portuguese politics. 

💾 The data is derived from [Debates Parlamentares](https://debates.parlamento.pt), for plenary speeches and the [Assembleia da República](https://www.parlamento.pt/Cidadania/Paginas/DAatividadeDeputado.aspx) for deputies' activities (such as questions to the government, law proposals and other parliamentary initiatives).

⌨️ There are three main components in this repository:

1. *SymmachusCore* defines a custom document embedding procedure based on word position and syntactic importance. That procedure is used to filter the data, according to specific speech acts, especially those stating explicit or implicit political intentions. 

2. *SymmachusModel* implements a self-training data set, where labels are based on linguistic expressions of *modality*. A semi-supervised approach is used, where a fraction of the total data set is manually labelled. This data is used to train an `XGBoost.jl` classifier. 

3. A semantic search engine on the basis of `sentence-transformers`. Two functionalities are supported by the backend: (a) Name Search – Searching an individual politician's corpus for matching political issues mentioned in speeches with activities undertaken and (b) Topic Search – Computing topic alignment scores for politicians and political parties. This can be any topic, but for the purpose of the thesis, the case study of SDGs (Sustainable Development Goals) is chosen.

**Note:** Stopwords used were accessed from this source https://members.unine.ch/jacques.savoy/clef/portugueseST2.txt, which unfortunately is currently unavailable.
